"""
Visualization utilities for validation benchmarks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def plot_side_by_side(
    solution1: NDArray,
    solution2: NDArray,
    title1: str = "Solution 1",
    title2: str = "Solution 2",
    figsize: tuple[int, int] = (12, 5),
) -> tuple:
    """
    Plot two 2D solutions side-by-side.

    Parameters
    ----------
    solution1, solution2 : NDArray
        2D solutions to plot
    title1, title2 : str
        Titles for each subplot
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes
        Matplotlib figure and axes
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Solution 1
    im1 = axes[0].imshow(solution1, origin="lower", cmap="viridis")
    axes[0].set_title(title1)
    plt.colorbar(im1, ax=axes[0])

    # Solution 2
    im2 = axes[1].imshow(solution2, origin="lower", cmap="viridis")
    axes[1].set_title(title2)
    plt.colorbar(im2, ax=axes[1])

    # Difference
    diff = np.abs(solution1 - solution2)
    im3 = axes[2].imshow(diff, origin="lower", cmap="hot")
    axes[2].set_title("Absolute Difference")
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    return fig, axes


def plot_convergence_comparison(
    residuals1: NDArray,
    residuals2: NDArray,
    label1: str = "Solver 1",
    label2: str = "Solver 2",
    figsize: tuple[int, int] = (10, 6),
) -> tuple:
    """
    Plot convergence comparison.

    Parameters
    ----------
    residuals1, residuals2 : NDArray
        Residual history for each solver
    label1, label2 : str
        Labels for each solver
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax
        Matplotlib figure and axis
    """
    fig, ax = plt.subplots(figsize=figsize)

    iterations1 = np.arange(len(residuals1))
    iterations2 = np.arange(len(residuals2))

    ax.semilogy(iterations1, residuals1, "o-", label=label1, linewidth=2)
    ax.semilogy(iterations2, residuals2, "s-", label=label2, linewidth=2)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Residual (log scale)", fontsize=12)
    ax.set_title("Convergence Comparison", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax
