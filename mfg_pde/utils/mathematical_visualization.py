#!/usr/bin/env python3
"""
Mathematical Visualization Module for MFG_PDE

Specialized visualization tools for mathematical analysis with comprehensive LaTeX
support for professional mathematical communication. Designed for mathematical 
researchers with emphasis on precise notation and publication-quality output.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Plotly imports with LaTeX support
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Matplotlib imports with LaTeX configuration
try:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from mpl_toolkits.mplot3d import Axes3D

    # Configure matplotlib for cross-platform compatibility
    rcParams["text.usetex"] = False  # Avoid LaTeX dependency
    rcParams["font.family"] = "sans-serif"  # Use system sans-serif fonts
    rcParams["font.sans-serif"] = [
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
        "Helvetica",
        "sans-serif",
    ]
    rcParams["mathtext.fontset"] = "dejavusans"  # Use DejaVu for math text
    rcParams["axes.formatter.use_mathtext"] = True

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .logging import get_logger


class MathematicalVisualizationError(Exception):
    """Exception for mathematical visualization errors."""

    pass


class MFGMathematicalVisualizer:
    """
    Mathematical visualization class with comprehensive LaTeX support.

    Designed for professional mathematical analysis and publication-quality
    figures with precise mathematical notation throughout.
    """

    def __init__(self, backend: str = "auto", enable_latex: bool = True):
        """
        Initialize mathematical visualizer.

        Args:
            backend: Visualization backend ("plotly", "matplotlib", or "auto")
            enable_latex: Enable LaTeX rendering (requires LaTeX installation)
        """
        self.logger = get_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        self.backend = self._validate_backend(backend)
        self.enable_latex = enable_latex

        # Configure mathematical notation settings
        if enable_latex and MATPLOTLIB_AVAILABLE:
            try:
                # Use mathtext for reliable mathematical notation without LaTeX dependency
                rcParams["text.usetex"] = False
                rcParams["mathtext.default"] = "regular"
                rcParams["font.family"] = "serif"
                rcParams["font.serif"] = [
                    "Computer Modern Roman",
                    "DejaVu Serif",
                    "serif",
                ]
                self.logger.info("LaTeX rendering enabled with mathtext")
            except Exception as e:
                self.logger.warning(f"LaTeX configuration failed: {e}, using defaults")
                rcParams["text.usetex"] = False

        self.logger.info(
            f"Mathematical visualizer initialized with backend: {self.backend}"
        )

    def _validate_backend(self, backend: str) -> str:
        """Validate visualization backend."""
        if backend == "auto":
            if MATPLOTLIB_AVAILABLE:  # Prefer matplotlib for mathematical work
                return "matplotlib"
            elif PLOTLY_AVAILABLE:
                return "plotly"
            else:
                raise MathematicalVisualizationError(
                    "No visualization backend available"
                )
        elif backend == "plotly" and not PLOTLY_AVAILABLE:
            raise MathematicalVisualizationError("Plotly backend not available")
        elif backend == "matplotlib" and not MATPLOTLIB_AVAILABLE:
            raise MathematicalVisualizationError("Matplotlib backend not available")

        return backend

    def plot_hjb_analysis(
        self,
        U: np.ndarray,
        x_grid: np.ndarray,
        t_grid: np.ndarray,
        gradients: Optional[Dict[str, np.ndarray]] = None,
        title: str = r"Hamilton-Jacobi-Bellman Analysis",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Any:
        """
        Comprehensive HJB equation analysis visualization.

        Creates subplots showing:
        - Value function $u(t,x)$
        - Spatial gradient $\\frac{\\partial u}{\\partial x}$
        - Temporal gradient $\\frac{\\partial u}{\\partial t}$
        - HJB residual analysis

        Args:
            U: Value function array $(N_x \\times N_t)$
            x_grid: Spatial discretization $x \\in [x_{\\min}, x_{\\max}]$
            t_grid: Temporal discretization $t \\in [0, T]$
            gradients: Dictionary containing gradient arrays
            title: Plot title with LaTeX support
            save_path: Output file path
            show: Display plot flag

        Returns:
            Figure object
        """
        self.logger.info("Creating comprehensive HJB analysis visualization")

        if self.backend == "matplotlib":
            return self._plot_hjb_matplotlib(
                U, x_grid, t_grid, gradients, title, save_path, show
            )
        else:
            return self._plot_hjb_plotly(
                U, x_grid, t_grid, gradients, title, save_path, show
            )

    def _plot_hjb_matplotlib(
        self, U, x_grid, t_grid, gradients, title, save_path, show
    ):
        """Create HJB analysis using matplotlib with LaTeX."""
        fig = plt.figure(figsize=(16, 12))

        # Create meshgrids
        X, T = np.meshgrid(x_grid, t_grid)

        # Main value function plot
        ax1 = plt.subplot(2, 3, (1, 2))
        contour = ax1.contourf(X, T, U.T, levels=20, cmap="viridis")
        ax1.set_xlabel(r"$x$", fontsize=14)
        ax1.set_ylabel(r"$t$", fontsize=14)
        ax1.set_title(r"Value Function $u(t,x)$", fontsize=16)
        plt.colorbar(contour, ax=ax1, label=r"$u(t,x)$")

        # 3D surface plot
        ax2 = plt.subplot(2, 3, 3, projection="3d")
        surf = ax2.plot_surface(X, T, U.T, cmap="viridis", alpha=0.8)
        ax2.set_xlabel(r"$x$", fontsize=12)
        ax2.set_ylabel(r"$t$", fontsize=12)
        ax2.set_zlabel(r"$u(t,x)$", fontsize=12)
        ax2.set_title(r"3D Surface $u(t,x)$", fontsize=14)

        # Spatial gradient
        if gradients and "du_dx" in gradients:
            ax3 = plt.subplot(2, 3, 4)
            contour3 = ax3.contourf(
                X, T, gradients["du_dx"].T, levels=15, cmap="RdBu_r"
            )
            ax3.set_xlabel(r"$x$", fontsize=14)
            ax3.set_ylabel(r"$t$", fontsize=14)
            ax3.set_title(
                r"Spatial Gradient $\frac{\partial u}{\partial x}$", fontsize=16
            )
            plt.colorbar(contour3, ax=ax3, label=r"$U_x$")
        else:
            # Compute numerical gradient
            du_dx = np.gradient(U, x_grid, axis=0)
            ax3 = plt.subplot(2, 3, 4)
            contour3 = ax3.contourf(X, T, du_dx.T, levels=15, cmap="RdBu_r")
            ax3.set_xlabel(r"$x$", fontsize=14)
            ax3.set_ylabel(r"$t$", fontsize=14)
            ax3.set_title(
                r"Spatial Gradient $\frac{\partial u}{\partial x}$", fontsize=16
            )
            plt.colorbar(contour3, ax=ax3, label=r"$U_x$")

        # Temporal gradient
        if gradients and "du_dt" in gradients:
            ax4 = plt.subplot(2, 3, 5)
            contour4 = ax4.contourf(
                X, T, gradients["du_dt"].T, levels=15, cmap="plasma"
            )
            ax4.set_xlabel(r"$x$", fontsize=14)
            ax4.set_ylabel(r"$t$", fontsize=14)
            ax4.set_title(
                r"Temporal Gradient $\frac{\partial u}{\partial t}$", fontsize=16
            )
            plt.colorbar(contour4, ax=ax4, label=r"$U_t$")
        else:
            # Compute numerical gradient
            du_dt = np.gradient(U, t_grid, axis=1)
            ax4 = plt.subplot(2, 3, 5)
            contour4 = ax4.contourf(X, T, du_dt.T, levels=15, cmap="plasma")
            ax4.set_xlabel(r"$x$", fontsize=14)
            ax4.set_ylabel(r"$t$", fontsize=14)
            ax4.set_title(
                r"Temporal Gradient $\frac{\partial u}{\partial t}$", fontsize=16
            )
            plt.colorbar(contour4, ax=ax4, label=r"$U_t$")

        # Solution profiles at different times
        ax5 = plt.subplot(2, 3, 6)
        n_profiles = 5
        t_indices = np.linspace(0, len(t_grid) - 1, n_profiles, dtype=int)
        colors = plt.cm.viridis(np.linspace(0, 1, n_profiles))

        for i, t_idx in enumerate(t_indices):
            ax5.plot(
                x_grid,
                U[:, t_idx],
                color=colors[i],
                linewidth=2,
                label=f"$t = {t_grid[t_idx]:.2f}$",
            )

        ax5.set_xlabel(r"$x$", fontsize=14)
        ax5.set_ylabel(r"$u(t,x)$", fontsize=14)
        ax5.set_title(r"Solution Profiles $u(t,x)$", fontsize=16)
        ax5.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc="upper left")
        ax5.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=18, y=0.96)
        plt.tight_layout(rect=[0, 0.03, 0.85, 0.92])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"HJB analysis saved to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_fokker_planck_analysis(
        self,
        M: np.ndarray,
        x_grid: np.ndarray,
        t_grid: np.ndarray,
        flux: Optional[np.ndarray] = None,
        title: str = r"Fokker-Planck Analysis",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Any:
        """
        Comprehensive Fokker-Planck equation analysis.

        Visualizes:
        - Density evolution $m(t,x)$
        - Probability flux $J(t,x) = m(t,x) \\alpha^*(t,x) - \\sigma^2/2 \\frac{\\partial m}{\\partial x}$
        - Mass conservation analysis
        - Statistical moments evolution

        Args:
            M: Density function $(N_x \\times N_t)$
            x_grid: Spatial grid
            t_grid: Temporal grid
            flux: Probability flux (optional)
            title: Plot title
            save_path: Save path
            show: Display flag

        Returns:
            Figure object
        """
        self.logger.info("Creating comprehensive Fokker-Planck analysis")

        if self.backend == "matplotlib":
            return self._plot_fp_matplotlib(
                M, x_grid, t_grid, flux, title, save_path, show
            )
        else:
            return self._plot_fp_plotly(M, x_grid, t_grid, flux, title, save_path, show)

    def _plot_fp_matplotlib(self, M, x_grid, t_grid, flux, title, save_path, show):
        """Fokker-Planck analysis with matplotlib."""
        fig = plt.figure(figsize=(16, 12))

        X, T = np.meshgrid(x_grid, t_grid)

        # Density evolution contour
        ax1 = plt.subplot(2, 3, 1)
        contour1 = ax1.contourf(X, T, M.T, levels=20, cmap="plasma")
        ax1.set_xlabel(r"$x$", fontsize=14)
        ax1.set_ylabel(r"$t$", fontsize=14)
        ax1.set_title(r"Density Evolution $m(t,x)$", fontsize=16)
        plt.colorbar(contour1, ax=ax1, label=r"$m(t,x)$")

        # 3D density surface
        ax2 = plt.subplot(2, 3, 2, projection="3d")
        surf2 = ax2.plot_surface(X, T, M.T, cmap="plasma", alpha=0.8)
        ax2.set_xlabel(r"$x$", fontsize=12)
        ax2.set_ylabel(r"$t$", fontsize=12)
        ax2.set_zlabel(r"$m(t,x)$", fontsize=12)
        ax2.set_title(r"3D Density $m(t,x)$", fontsize=14)

        # Mass conservation check
        ax3 = plt.subplot(2, 3, 3)
        total_mass = np.trapz(M, x_grid, axis=0)
        ax3.plot(t_grid, total_mass, "b-", linewidth=2, label=r"$\int m(t,x) dx$")
        ax3.axhline(y=1, color="r", linestyle="--", alpha=0.7, label=r"Expected: $1$")
        ax3.set_xlabel(r"$t$", fontsize=14)
        ax3.set_ylabel(r"Total Mass", fontsize=14)
        ax3.set_title(r"Mass Conservation $\int_{\Omega} m(t,x) dx$", fontsize=16)
        ax3.legend(fontsize=10, loc="upper right")
        ax3.grid(True, alpha=0.3)

        # Statistical moments evolution
        ax4 = plt.subplot(2, 3, 4)

        # Compute moments
        mean_x = np.trapz(x_grid[:, np.newaxis] * M, x_grid, axis=0)
        var_x = np.trapz(
            (x_grid[:, np.newaxis] - mean_x[np.newaxis, :]) ** 2 * M, x_grid, axis=0
        )

        ax4.plot(
            t_grid,
            mean_x,
            "b-",
            linewidth=2,
            label=r"$\mathbb{E}[X_t] = \int x m(t,x) dx$",
        )
        ax4_twin = ax4.twinx()
        ax4_twin.plot(
            t_grid,
            var_x,
            "r-",
            linewidth=2,
            label=r"$\mathrm{Var}[X_t] = \int (x-\mu_t)^2 m(t,x) dx$",
        )

        ax4.set_xlabel(r"$t$", fontsize=14)
        ax4.set_ylabel(r"Mean $\mu_t$", fontsize=14, color="b")
        ax4_twin.set_ylabel(r"Variance $\sigma_t^2$", fontsize=14, color="r")
        ax4.set_title(r"Statistical Moments Evolution", fontsize=16)
        ax4.grid(True, alpha=0.3)

        # Combine legends - place outside to avoid overlap
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(
            lines1 + lines2,
            labels1 + labels2,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=9,
        )

        # Density profiles
        ax5 = plt.subplot(2, 3, 5)
        n_profiles = 5
        t_indices = np.linspace(0, len(t_grid) - 1, n_profiles, dtype=int)
        colors = plt.cm.plasma(np.linspace(0, 1, n_profiles))

        for i, t_idx in enumerate(t_indices):
            ax5.plot(
                x_grid,
                M[:, t_idx],
                color=colors[i],
                linewidth=2,
                label=f"$t = {t_grid[t_idx]:.2f}$",
            )

        ax5.set_xlabel(r"$x$", fontsize=14)
        ax5.set_ylabel(r"$m(t,x)$", fontsize=14)
        ax5.set_title(r"Density Profiles $m(t,x)$", fontsize=16)
        ax5.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc="upper left")
        ax5.grid(True, alpha=0.3)

        # Flux analysis (if provided)
        if flux is not None:
            ax6 = plt.subplot(2, 3, 6)
            contour6 = ax6.contourf(X, T, flux.T, levels=15, cmap="RdBu_r")
            ax6.set_xlabel(r"$x$", fontsize=14)
            ax6.set_ylabel(r"$t$", fontsize=14)
            ax6.set_title(r"Probability Flux $J(x,t)$", fontsize=16)
            plt.colorbar(contour6, ax=ax6, label=r"$J(x,t)$")
        else:
            # Compute flux divergence for continuity equation check
            ax6 = plt.subplot(2, 3, 6)
            dm_dt = np.gradient(M, t_grid, axis=1)
            div_flux = -dm_dt  # From continuity equation: ∂m/∂t + ∇·J = 0

            contour6 = ax6.contourf(X, T, div_flux.T, levels=15, cmap="RdBu_r")
            ax6.set_xlabel(r"$x$", fontsize=14)
            ax6.set_ylabel(r"$t$", fontsize=14)
            ax6.set_title(
                r"Flux Divergence $-\frac{\partial m}{\partial t}$", fontsize=16
            )
            plt.colorbar(contour6, ax=ax6, label=r"$\nabla \cdot J$")

        plt.suptitle(title, fontsize=18, y=0.96)
        plt.tight_layout(rect=[0, 0.03, 0.85, 0.92])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Fokker-Planck analysis saved to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_convergence_theory(
        self,
        convergence_data: Dict[str, List[float]],
        theoretical_rates: Optional[Dict[str, float]] = None,
        title: str = r"Convergence Analysis",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Any:
        """
        Theoretical convergence analysis with rate estimation.

        Plots convergence history with theoretical convergence rates and
        estimates actual convergence rates using regression analysis.

        Args:
            convergence_data: Convergence metrics
            theoretical_rates: Expected convergence rates $\\{\\rho_k\\}$
            title: Plot title
            save_path: Save path
            show: Display flag

        Returns:
            Figure object
        """
        self.logger.info("Creating theoretical convergence analysis")

        if self.backend == "matplotlib":
            return self._plot_convergence_theory_matplotlib(
                convergence_data, theoretical_rates, title, save_path, show
            )
        else:
            return self._plot_convergence_theory_plotly(
                convergence_data, theoretical_rates, title, save_path, show
            )

    def _plot_convergence_theory_matplotlib(
        self, convergence_data, theoretical_rates, title, save_path, show
    ):
        """Convergence theory analysis with matplotlib."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Main convergence plot (log scale)
        ax1 = axes[0, 0]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for i, (metric, values) in enumerate(convergence_data.items()):
            iterations = np.arange(1, len(values) + 1)
            color = colors[i % len(colors)]

            ax1.semilogy(
                iterations,
                values,
                "o-",
                color=color,
                linewidth=2,
                label=metric,
                markersize=4,
            )

            # Add theoretical rate if provided
            if theoretical_rates and metric in theoretical_rates:
                rate = theoretical_rates[metric]
                theoretical = values[0] * (rate ** (iterations - 1))
                ax1.semilogy(
                    iterations,
                    theoretical,
                    "--",
                    color=color,
                    alpha=0.7,
                    label=f"{metric} (rate: {rate:.3f})",
                )

        ax1.set_xlabel(r"Iteration $k$", fontsize=14)
        ax1.set_ylabel(r"Error $\|e_k\|$", fontsize=14)
        ax1.set_title(r"Convergence History", fontsize=16)
        ax1.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Convergence rate estimation
        ax2 = axes[0, 1]

        for i, (metric, values) in enumerate(convergence_data.items()):
            if len(values) > 5:  # Need sufficient data points
                # Estimate convergence rate: log(e_{k+1}) ≈ log(ρ) + log(e_k)
                log_errors = np.log(values[:-1])
                log_errors_next = np.log(values[1:])

                # Linear regression
                coeffs = np.polyfit(log_errors, log_errors_next, 1)
                estimated_rate = np.exp(coeffs[0])

                iterations = np.arange(1, len(values))
                predicted = coeffs[1] + coeffs[0] * log_errors

                ax2.plot(
                    log_errors,
                    log_errors_next,
                    "o",
                    color=colors[i % len(colors)],
                    label=f"{metric}: $\\rho \\approx {estimated_rate:.3f}$",
                )
                ax2.plot(
                    log_errors, predicted, "-", color=colors[i % len(colors)], alpha=0.7
                )

        ax2.set_xlabel(r"$\log(\|e_k\|)$", fontsize=14)
        ax2.set_ylabel(r"$\log(\|e_{k+1}\|)$", fontsize=14)
        ax2.set_title(
            r"Rate Estimation: $\log(e_{k+1}) \approx \rho \log(e_k)$", fontsize=16
        )
        ax2.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(True, alpha=0.3)

        # Residual analysis
        ax3 = axes[1, 0]

        for i, (metric, values) in enumerate(convergence_data.items()):
            if len(values) > 2:
                # Compute discrete derivative (convergence rate)
                rates = np.array(values[1:]) / np.array(values[:-1])
                iterations = np.arange(2, len(values) + 1)

                ax3.plot(
                    iterations,
                    rates,
                    "o-",
                    color=colors[i % len(colors)],
                    label=f"{metric}: $e_k/e_{{k-1}}$",
                    linewidth=2,
                    markersize=4,
                )

        ax3.set_xlabel(r"Iteration $k$", fontsize=14)
        ax3.set_ylabel(r"Convergence Factor $\rho_k = e_k/e_{k-1}$", fontsize=14)
        ax3.set_title(r"Instantaneous Convergence Rates", fontsize=16)
        ax3.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc="upper left")
        ax3.grid(True, alpha=0.3)

        # Error reduction per iteration
        ax4 = axes[1, 1]

        for i, (metric, values) in enumerate(convergence_data.items()):
            if len(values) > 1:
                reductions = -np.diff(np.log(values))  # -log(e_{k+1}/e_k)
                iterations = np.arange(2, len(values) + 1)

                ax4.plot(
                    iterations,
                    reductions,
                    "s-",
                    color=colors[i % len(colors)],
                    label=f"{metric}",
                    linewidth=2,
                    markersize=4,
                )

        ax4.set_xlabel(r"Iteration $k$", fontsize=14)
        ax4.set_ylabel(r"Error Reduction $-\log(e_k/e_{k-1})$", fontsize=14)
        ax4.set_title(r"Logarithmic Error Reduction", fontsize=16)
        ax4.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc="upper left")
        ax4.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=18, y=0.96)
        plt.tight_layout(rect=[0, 0.03, 0.85, 0.92])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Convergence theory analysis saved to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_phase_space_analysis(
        self,
        U: np.ndarray,
        M: np.ndarray,
        x_grid: np.ndarray,
        t_grid: np.ndarray,
        title: str = r"Phase Space Analysis",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Any:
        """
        Phase space analysis of the MFG system.

        Visualizes the dynamics in the $(U, m)$ phase space and analyzes
        the coupled evolution of value function and density.

        Args:
            U: Value function
            M: Density function
            x_grid: Spatial grid
            t_grid: Temporal grid
            title: Plot title
            save_path: Save path
            show: Display flag

        Returns:
            Figure object
        """
        self.logger.info("Creating phase space analysis")

        if not MATPLOTLIB_AVAILABLE:
            raise MathematicalVisualizationError(
                "Phase space analysis requires matplotlib"
            )

        fig = plt.figure(figsize=(16, 12))

        # Phase portrait in (u, m) space
        ax1 = plt.subplot(2, 3, 1)

        # Sample points for phase portrait
        n_sample = min(20, len(x_grid))
        x_indices = np.linspace(0, len(x_grid) - 1, n_sample, dtype=int)

        colors = plt.cm.viridis(np.linspace(0, 1, len(t_grid)))

        for i, x_idx in enumerate(x_indices[::2]):  # Every other point to avoid clutter
            U_traj = U[x_idx, :]
            M_traj = M[x_idx, :]

            ax1.plot(U_traj, M_traj, "-", alpha=0.6, linewidth=1)

            # Mark time evolution with colors
            for t_idx in range(0, len(t_grid), max(1, len(t_grid) // 10)):
                ax1.plot(
                    U_traj[t_idx], M_traj[t_idx], "o", color=colors[t_idx], markersize=3
                )

        ax1.set_xlabel(r"Value Function $u(t,x)$", fontsize=14)
        ax1.set_ylabel(r"Density $m(t,x)$", fontsize=14)
        ax1.set_title(r"Phase Portrait $(u, m)$", fontsize=16)
        ax1.grid(True, alpha=0.3)

        # Energy functional evolution
        ax2 = plt.subplot(2, 3, 2)

        # Compute energy: E(t) = ∫ [½|∇U|² + f(x,U,m)] m dx
        energy = np.zeros(len(t_grid))
        for t_idx in range(len(t_grid)):
            grad_U = np.gradient(U[:, t_idx], x_grid)
            integrand = 0.5 * grad_U**2 * M[:, t_idx]
            energy[t_idx] = np.trapz(integrand, x_grid)

        ax2.plot(
            t_grid,
            energy,
            "b-",
            linewidth=2,
            label=r"$E(t) = \int \frac{1}{2}|\nabla u|^2 m \, dx$",
        )
        ax2.set_xlabel(r"Time $t$", fontsize=14)
        ax2.set_ylabel(r"Energy $E(t)$", fontsize=14)
        ax2.set_title(r"Energy Functional Evolution", fontsize=16)
        ax2.legend(fontsize=10, loc="upper right")
        ax2.grid(True, alpha=0.3)

        # Hamiltonian density
        ax3 = plt.subplot(2, 3, 3)

        X, T = np.meshgrid(x_grid, t_grid)

        # Compute Hamiltonian H(x, ∇U, m)
        H = np.zeros_like(U.T)
        for t_idx in range(len(t_grid)):
            grad_U = np.gradient(U[:, t_idx], x_grid)
            H[t_idx, :] = 0.5 * grad_U**2  # H(x, p, m) = ½p²

        contour3 = ax3.contourf(X, T, H, levels=15, cmap="plasma")
        ax3.set_xlabel(r"$x$", fontsize=14)
        ax3.set_ylabel(r"$t$", fontsize=14)
        ax3.set_title(r"Hamiltonian Density $H(x, \nabla u, m)$", fontsize=16)
        plt.colorbar(contour3, ax=ax3, label=r"$H$")

        # Optimal control
        ax4 = plt.subplot(2, 3, 4)

        # Compute optimal control α* = -∇H/∇p = -∇U
        alpha_star = np.zeros_like(U)
        for t_idx in range(len(t_grid)):
            alpha_star[:, t_idx] = -np.gradient(U[:, t_idx], x_grid)

        contour4 = ax4.contourf(X, T, alpha_star.T, levels=15, cmap="RdBu_r")
        ax4.set_xlabel(r"$x$", fontsize=14)
        ax4.set_ylabel(r"$t$", fontsize=14)
        ax4.set_title(
            r"Optimal Control $\alpha^*(x,t) = -\frac{\partial H}{\partial p}$",
            fontsize=16,
        )
        plt.colorbar(contour4, ax=ax4, label=r"$\alpha^*$")

        # Lagrangian analysis
        ax5 = plt.subplot(2, 3, 5)

        # Action integral approximation
        action = np.zeros(len(t_grid))
        for t_idx in range(len(t_grid)):
            # L = ∫ m(x,t) [α²/2 + f(x,U,m)] dx (simplified)
            alpha = alpha_star[:, t_idx]
            lagrangian_density = M[:, t_idx] * (0.5 * alpha**2)
            action[t_idx] = np.trapz(lagrangian_density, x_grid)

        ax5.plot(
            t_grid,
            action,
            "g-",
            linewidth=2,
            label=r"$L(t) = \int m \frac{|\alpha|^2}{2} \, dx$",
        )
        ax5.set_xlabel(r"Time $t$", fontsize=14)
        ax5.set_ylabel(r"Action $L(t)$", fontsize=14)
        ax5.set_title(r"Lagrangian Evolution", fontsize=16)
        ax5.legend(fontsize=10, loc="upper right")
        ax5.grid(True, alpha=0.3)

        # Pontryagin maximum principle check
        ax6 = plt.subplot(2, 3, 6)

        # Compute costate evolution (simplified)
        costate = np.zeros_like(M)
        for x_idx in range(len(x_grid)):
            # Costate ≈ ∂L/∂m (simplified analysis)
            costate[x_idx, :] = U[x_idx, :] - np.mean(U[x_idx, :])

        # Plot costate at different positions
        for i, x_idx in enumerate(range(0, len(x_grid), len(x_grid) // 5)):
            ax6.plot(
                t_grid,
                costate[x_idx, :],
                linewidth=2,
                label=f"$x = {x_grid[x_idx]:.2f}$",
            )

        ax6.set_xlabel(r"Time $t$", fontsize=14)
        ax6.set_ylabel(r"Costate $\lambda(t,x)$", fontsize=14)
        ax6.set_title(r"Costate Evolution", fontsize=16)
        ax6.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc="upper left")
        ax6.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=18, y=0.96)
        plt.tight_layout(rect=[0, 0.03, 0.85, 0.92])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Phase space analysis saved to {save_path}")

        if show:
            plt.show()

        return fig


# Convenience functions with LaTeX support
def quick_hjb_analysis(
    U: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    save_path: Optional[str] = None,
) -> Any:
    """Quick HJB analysis with LaTeX mathematical notation."""
    visualizer = MFGMathematicalVisualizer(backend="auto", enable_latex=True)
    return visualizer.plot_hjb_analysis(U, x_grid, t_grid, save_path=save_path)


def quick_fp_analysis(
    M: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    save_path: Optional[str] = None,
) -> Any:
    """Quick Fokker-Planck analysis with LaTeX mathematical notation."""
    visualizer = MFGMathematicalVisualizer(backend="auto", enable_latex=True)
    return visualizer.plot_fokker_planck_analysis(
        M, x_grid, t_grid, save_path=save_path
    )


def quick_phase_space_analysis(
    U: np.ndarray,
    M: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    save_path: Optional[str] = None,
) -> Any:
    """Quick phase space analysis with LaTeX mathematical notation."""
    visualizer = MFGMathematicalVisualizer(backend="auto", enable_latex=True)
    return visualizer.plot_phase_space_analysis(
        U, M, x_grid, t_grid, save_path=save_path
    )


# Example and demonstration
def demo_mathematical_visualization():
    """Demonstrate mathematical visualization capabilities with LaTeX."""
    logger = get_logger("mfg_pde.utils.mathematical_visualization")
    logger.info("Starting mathematical visualization demonstration")

    # Generate sample mathematical data
    x_grid = np.linspace(0, 1, 50)
    t_grid = np.linspace(0, 1, 25)
    X, T = np.meshgrid(x_grid, t_grid)

    # Sample solutions with mathematical structure
    U = np.sin(np.pi * X) * np.exp(-T) + 0.1 * np.sin(3 * np.pi * X) * np.exp(-2 * T)
    M = np.exp(-10 * (X - 0.3 - 0.2 * T) ** 2) + np.exp(-10 * (X - 0.7 + 0.1 * T) ** 2)

    # Normalize density
    for i in range(len(t_grid)):
        M[:, i] = M[:, i] / np.trapz(M[:, i], x_grid)

    U = U.T
    M = M.T

    try:
        visualizer = MFGMathematicalVisualizer(backend="matplotlib", enable_latex=True)

        # Test HJB analysis
        hjb_fig = visualizer.plot_hjb_analysis(
            U,
            x_grid,
            t_grid,
            title=r"HJB Analysis: $-\frac{\partial u}{\partial t} + H(x, \nabla u, m) = 0$",
            show=False,
        )

        # Test FP analysis
        fp_fig = visualizer.plot_fokker_planck_analysis(
            M,
            x_grid,
            t_grid,
            title=r"FP Analysis: $\frac{\partial m}{\partial t} - \nabla \cdot (m \nabla H_p) - \frac{\sigma^2}{2}\Delta m = 0$",
            show=False,
        )

        # Test convergence theory
        convergence_data = {
            r"$\|U^k - U^*\|_{L^2}$": [1e-1 * (0.8**i) for i in range(20)],
            r"$\|m^k - m^*\|_{L^1}$": [5e-2 * (0.85**i) for i in range(20)],
        }

        conv_fig = visualizer.plot_convergence_theory(
            convergence_data,
            theoretical_rates={
                r"$\|U^k - U^*\|_{L^2}$": 0.8,
                r"$\|m^k - m^*\|_{L^1}$": 0.85,
            },
            title=r"Convergence Theory: $\|e^k\| \leq C \rho^k \|e^0\|$",
            show=False,
        )

        # Test phase space analysis
        phase_fig = visualizer.plot_phase_space_analysis(
            U,
            M,
            x_grid,
            t_grid,
            title=r"Phase Space: Hamiltonian System $(u, m)$",
            show=False,
        )

        logger.info("Mathematical visualization demonstration completed successfully")

    except Exception as e:
        logger.error(f"Error in mathematical visualization: {e}")
        raise


if __name__ == "__main__":
    demo_mathematical_visualization()
