"""
Basic matplotlib plotting functions for MFG solutions.

Provides simple, no-dependency visualization for solver development:
- 3D surface plots of u(t,x) and m(t,x)
- Convergence history plots
- Final density and mass conservation plots

For interactive or publication-quality visualization, use Plotly/matplotlib directly.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator


def myplot3d(X, Y, Z, title="Surface Plot"):
    """Create 3D surface plot with matplotlib."""
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.get_cmap("coolwarm"), linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    ax.set_xlabel("x")
    ax.set_ylabel("time")
    ax.set_title(title)
    ax.view_init(40, -135)
    plt.show()


def plot_convergence(iterations_run, l2disturel_u, l2disturel_m, solver_name="Solver"):
    """Plot convergence history for U and M."""
    iterSpace = np.arange(1, iterations_run + 1)

    plt.figure()
    plt.semilogy(iterSpace, l2disturel_u)
    plt.xlabel("Iteration")
    plt.ylabel("$||u_{new}-u_{old}||_{rel}$")
    plt.title(f"Convergence of U ({solver_name})")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.semilogy(iterSpace, l2disturel_m)
    plt.xlabel("Iteration")
    plt.ylabel("$||m_{new}-m_{old}||_{rel}$")
    plt.title(f"Convergence of M ({solver_name})")
    plt.grid(True)
    plt.show()


def plot_results(problem, u, m, solver_name="Solver", prefix=None):
    """Plot MFG results: 3D surfaces of u and m, final density, mass conservation."""
    xSpace = problem.geometry.coordinates[0]
    tSpace = problem.tSpace
    dx = problem.geometry.get_grid_spacing()[0]

    kx = 2
    kt = 5
    xSpacecut = xSpace[::kx]
    tSpacecut = tSpace[::kt]
    ucut = u[::kt, ::kx]
    mcut = m[::kt, ::kx]

    myplot3d(xSpacecut, tSpacecut, ucut, title=f"Evolution of U ({solver_name})")
    myplot3d(xSpacecut, tSpacecut, mcut, title=f"Evolution of M ({solver_name})")

    plt.figure()
    plt.plot(xSpace, m[-1, :])
    plt.xlabel("x")
    plt.ylabel("m(T,x)")
    plt.title(f"Final Density m(T,x) ({solver_name})")
    plt.grid(True)
    plt.show()

    plt.figure()
    mtot = np.sum(m * dx, axis=1)
    plt.plot(tSpace, mtot)
    plt.xlabel("t")
    plt.ylabel("Total Mass $\\int m(t)$")
    plt.title(f"Total Mass ({solver_name})")
    plt.ylim(
        min(0.9, np.min(mtot) * 0.98 if mtot.size > 0 else 0.9),
        max(1.1, np.max(mtot) * 1.02 if mtot.size > 0 else 1.1),
    )
    plt.grid(True)
    plt.show()
