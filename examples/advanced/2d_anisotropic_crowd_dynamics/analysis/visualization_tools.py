"""
Visualization Tools for 2D Anisotropic Crowd Dynamics

This module provides comprehensive visualization capabilities for analyzing
anisotropic MFG solutions, including density evolution, velocity fields,
barrier effects, and comparative analysis.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Using matplotlib only.")

# Local imports
import sys

from mfg_pde.utils.logging import get_logger

sys.path.append("..")

logger = get_logger(__name__)


class AnisotropicVisualizer:
    """
    Comprehensive visualization suite for anisotropic crowd dynamics.

    Provides both static (matplotlib) and interactive (plotly) visualizations
    for analyzing MFG solutions with barriers and anisotropic effects.
    """

    def __init__(self, problem: AnisotropicCrowdDynamics, solution: Any):
        """
        Initialize visualizer with problem and solution.

        Args:
            problem: AnisotropicCrowdDynamics problem instance
            solution: Solved MFG solution
        """
        self.problem = problem
        self.solution = solution
        self.domain = problem.domain_bounds
        self.has_barriers = len(problem.barriers) > 0

        # Create coordinate grids
        nx, ny = problem.grid_size
        self.x = np.linspace(self.domain[0][0], self.domain[0][1], nx)
        self.y = np.linspace(self.domain[1][0], self.domain[1][1], ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        logger.info(f"Initialized visualizer: grid={problem.grid_size}, barriers={self.has_barriers}")

    def create_density_evolution_animation(
        self, output_file: str = "density_evolution.html", time_indices: list[int] | None = None
    ):
        """
        Create animated visualization of density evolution.

        Args:
            output_file: Output filename
            time_indices: Specific time indices to include (None for all)
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Creating static snapshots instead.")
            return self._create_density_snapshots_matplotlib(time_indices)

        logger.info("Creating density evolution animation")

        if time_indices is None:
            time_indices = list(
                range(0, len(self.solution.density_history), max(1, len(self.solution.density_history) // 20))
            )

        # Create frames for animation
        frames = []
        for i in time_indices:
            m = self.solution.density_history[i]
            t = self.solution.time_grid[i] if hasattr(self.solution, "time_grid") else i * 0.01

            frame = go.Frame(
                data=[
                    go.Heatmap(
                        x=self.x,
                        y=self.y,
                        z=m,
                        colorscale="Viridis",
                        zmin=0,
                        zmax=np.max([np.max(m) for m in self.solution.density_history]),
                        colorbar=dict(title="Density"),
                    )
                ],
                name=f"t={t:.3f}",
            )
            frames.append(frame)

        # Initial frame
        initial_density = self.solution.density_history[0]

        fig = go.Figure(
            data=[
                go.Heatmap(x=self.x, y=self.y, z=initial_density, colorscale="Viridis", colorbar=dict(title="Density"))
            ],
            frames=frames,
        )

        # Add barriers if present
        if self.has_barriers:
            self._add_barriers_to_plotly_figure(fig)

        # Add animation controls
        fig.update_layout(
            title="2D Anisotropic Crowd Dynamics: Density Evolution",
            xaxis_title="x₁",
            yaxis_title="x₂",
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True}],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
        )

        fig.write_html(output_file)
        logger.info(f"Density animation saved to {output_file}")

    def create_velocity_field_plot(
        self, time_index: int = -1, output_file: str = "velocity_field.png", subsample: int = 4
    ):
        """
        Create velocity field visualization.

        Args:
            time_index: Time index to visualize (-1 for final time)
            output_file: Output filename
            subsample: Subsampling factor for velocity vectors
        """
        logger.info(f"Creating velocity field plot at time index {time_index}")

        # Get density and value function
        m = self.solution.density_history[time_index]
        u = self.solution.value_history[time_index]

        # Compute velocity field
        grad_u = np.gradient(u)
        grid_points = np.column_stack([self.X.ravel(), self.Y.ravel()])
        velocity = -self.problem.compute_hamiltonian_gradient(
            grid_points,
            np.column_stack([grad_u[1].ravel(), grad_u[0].ravel()]),  # Note: gradient order
            m.ravel(),
        )

        # Reshape velocity components
        vx = velocity[:, 0].reshape(self.X.shape)
        vy = velocity[:, 1].reshape(self.Y.shape)

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Density plot
        im1 = ax1.contourf(self.X, self.Y, m, levels=20, cmap="viridis")
        ax1.set_title("Density Distribution")
        ax1.set_xlabel("x₁")
        ax1.set_ylabel("x₂")
        plt.colorbar(im1, ax=ax1, label="Density")

        # Velocity field plot
        # Subsample for cleaner visualization
        X_sub = self.X[::subsample, ::subsample]
        Y_sub = self.Y[::subsample, ::subsample]
        vx_sub = vx[::subsample, ::subsample]
        vy_sub = vy[::subsample, ::subsample]

        # Plot density as background
        ax2.contourf(self.X, self.Y, m, levels=20, cmap="viridis", alpha=0.5)

        # Plot velocity vectors
        ax2.quiver(
            X_sub,
            Y_sub,
            vx_sub,
            vy_sub,
            scale=None,
            scale_units="xy",
            angles="xy",
            color="white",
            alpha=0.8,
            width=0.003,
        )

        ax2.set_title("Velocity Field")
        ax2.set_xlabel("x₁")
        ax2.set_ylabel("x₂")

        # Add barriers if present
        if self.has_barriers:
            self._add_barriers_to_matplotlib_axes([ax1, ax2])

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Velocity field plot saved to {output_file}")

    def create_anisotropy_analysis_plot(self, output_file: str = "anisotropy_analysis.png"):
        """
        Create visualization showing anisotropy function and its effects.

        Args:
            output_file: Output filename
        """
        logger.info("Creating anisotropy analysis plot")

        # Compute anisotropy function
        grid_points = np.column_stack([self.X.ravel(), self.Y.ravel()])
        rho = self.problem.compute_anisotropy(grid_points).reshape(self.X.shape)

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Anisotropy function
        im1 = axes[0, 0].contourf(self.X, self.Y, rho, levels=20, cmap="RdBu_r")
        axes[0, 0].set_title("Anisotropy Function ρ(x)")
        axes[0, 0].set_xlabel("x₁")
        axes[0, 0].set_ylabel("x₂")
        plt.colorbar(im1, ax=axes[0, 0])

        # Initial density
        m_initial = self.solution.density_history[0]
        im2 = axes[0, 1].contourf(self.X, self.Y, m_initial, levels=20, cmap="viridis")
        axes[0, 1].set_title("Initial Density")
        axes[0, 1].set_xlabel("x₁")
        axes[0, 1].set_ylabel("x₂")
        plt.colorbar(im2, ax=axes[0, 1])

        # Final density
        m_final = self.solution.density_history[-1]
        im3 = axes[1, 0].contourf(self.X, self.Y, m_final, levels=20, cmap="viridis")
        axes[1, 0].set_title("Final Density")
        axes[1, 0].set_xlabel("x₁")
        axes[1, 0].set_ylabel("x₂")
        plt.colorbar(im3, ax=axes[1, 0])

        # Density difference (final - initial)
        density_change = m_final - m_initial
        im4 = axes[1, 1].contourf(self.X, self.Y, density_change, levels=20, cmap="RdBu_r")
        axes[1, 1].set_title("Density Change")
        axes[1, 1].set_xlabel("x₁")
        axes[1, 1].set_ylabel("x₂")
        plt.colorbar(im4, ax=axes[1, 1])

        # Add barriers if present
        if self.has_barriers:
            self._add_barriers_to_matplotlib_axes(axes.ravel())

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Anisotropy analysis plot saved to {output_file}")

    def create_barrier_influence_analysis(self, output_file: str = "barrier_influence.png"):
        """
        Analyze and visualize barrier influence on flow patterns.

        Args:
            output_file: Output filename
        """
        if not self.has_barriers:
            logger.warning("No barriers present. Skipping barrier influence analysis.")
            return

        logger.info("Creating barrier influence analysis")

        # Get final density and velocity
        m_final = self.solution.density_history[-1]
        u_final = self.solution.value_history[-1]

        # Compute velocity field
        grad_u = np.gradient(u_final)
        grid_points = np.column_stack([self.X.ravel(), self.Y.ravel()])
        velocity = -self.problem.compute_hamiltonian_gradient(
            grid_points, np.column_stack([grad_u[1].ravel(), grad_u[0].ravel()]), m_final.ravel()
        )

        # Compute circulation around barriers
        vx = velocity[:, 0].reshape(self.X.shape)
        vy = velocity[:, 1].reshape(self.Y.shape)
        circulation = np.gradient(vy, axis=1) - np.gradient(vx, axis=0)

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Density with streamlines
        axes[0].contourf(self.X, self.Y, m_final, levels=20, cmap="viridis", alpha=0.7)
        axes[0].streamplot(self.X, self.Y, vx, vy, color="white", density=2, linewidth=1)
        axes[0].set_title("Flow Streamlines")
        axes[0].set_xlabel("x₁")
        axes[0].set_ylabel("x₂")

        # Velocity magnitude
        vel_magnitude = np.sqrt(vx**2 + vy**2)
        im2 = axes[1].contourf(self.X, self.Y, vel_magnitude, levels=20, cmap="plasma")
        axes[1].set_title("Velocity Magnitude")
        axes[1].set_xlabel("x₁")
        axes[1].set_ylabel("x₂")
        plt.colorbar(im2, ax=axes[1])

        # Circulation/vorticity
        im3 = axes[2].contourf(self.X, self.Y, circulation, levels=20, cmap="RdBu_r")
        axes[2].set_title("Circulation")
        axes[2].set_xlabel("x₁")
        axes[2].set_ylabel("x₂")
        plt.colorbar(im3, ax=axes[2])

        # Add barriers
        self._add_barriers_to_matplotlib_axes(axes)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Barrier influence analysis saved to {output_file}")

    def create_evacuation_metrics_dashboard(self, output_file: str = "metrics_dashboard.html"):
        """
        Create interactive dashboard showing evacuation metrics.

        Args:
            output_file: Output filename
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Creating static metrics plot instead.")
            return self._create_metrics_plot_matplotlib(output_file.replace(".html", ".png"))

        logger.info("Creating evacuation metrics dashboard")

        # Compute metrics over time
        time_grid = getattr(
            self.solution, "time_grid", np.linspace(0, self.problem.time_horizon, len(self.solution.density_history))
        )

        total_mass = [np.sum(m) for m in self.solution.density_history]
        peak_density = [np.max(m) for m in self.solution.density_history]

        # Compute evacuation percentage
        initial_mass = total_mass[0]
        evacuation_percentage = [(initial_mass - mass) / initial_mass * 100 for mass in total_mass]

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Total Mass", "Peak Density", "Evacuation Progress", "Final Density Distribution"),
            specs=[[{"type": "scatter"}, {"type": "scatter"}], [{"type": "scatter"}, {"type": "heatmap"}]],
        )

        # Total mass over time
        fig.add_trace(go.Scatter(x=time_grid, y=total_mass, mode="lines", name="Total Mass"), row=1, col=1)

        # Peak density over time
        fig.add_trace(
            go.Scatter(x=time_grid, y=peak_density, mode="lines", name="Peak Density", line=dict(color="red")),
            row=1,
            col=2,
        )

        # Evacuation progress
        fig.add_trace(
            go.Scatter(
                x=time_grid, y=evacuation_percentage, mode="lines", name="Evacuated %", line=dict(color="green")
            ),
            row=2,
            col=1,
        )

        # Final density distribution
        fig.add_trace(
            go.Heatmap(x=self.x, y=self.y, z=self.solution.density_history[-1], colorscale="viridis"), row=2, col=2
        )

        fig.update_layout(title="Evacuation Metrics Dashboard", showlegend=False, height=600)

        fig.write_html(output_file)
        logger.info(f"Metrics dashboard saved to {output_file}")

    def create_comparative_analysis(
        self, other_solutions: dict[str, Any], output_file: str = "comparative_analysis.html"
    ):
        """
        Create comparative analysis between different configurations.

        Args:
            other_solutions: Dictionary of {label: solution} for comparison
            output_file: Output filename
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for comparative analysis.")
            return

        logger.info("Creating comparative analysis")

        # Include current solution
        all_solutions = {"Current": self.solution}
        all_solutions.update(other_solutions)

        # Create comparison plots
        fig = make_subplots(
            rows=2,
            cols=len(all_solutions),
            subplot_titles=[f"{label} - Final Density" for label in all_solutions]
            + [f"{label} - Evacuation" for label in all_solutions],
            specs=[[{"type": "heatmap"}] * len(all_solutions), [{"type": "scatter"}] * len(all_solutions)],
        )

        colors = px.colors.qualitative.Set1

        for i, (label, solution) in enumerate(all_solutions.items()):
            # Final density heatmap
            fig.add_trace(
                go.Heatmap(z=solution.density_history[-1], colorscale="viridis", showscale=(i == 0)), row=1, col=i + 1
            )

            # Evacuation curve
            time_grid = getattr(
                solution, "time_grid", np.linspace(0, self.problem.time_horizon, len(solution.density_history))
            )
            total_mass = [np.sum(m) for m in solution.density_history]
            initial_mass = total_mass[0]
            evacuation_percentage = [(initial_mass - mass) / initial_mass * 100 for mass in total_mass]

            fig.add_trace(
                go.Scatter(
                    x=time_grid,
                    y=evacuation_percentage,
                    mode="lines",
                    name=label,
                    line=dict(color=colors[i % len(colors)]),
                ),
                row=2,
                col=i + 1,
            )

        fig.update_layout(title="Comparative Analysis: Different Configurations", height=800)

        fig.write_html(output_file)
        logger.info(f"Comparative analysis saved to {output_file}")

    def _create_density_snapshots_matplotlib(self, time_indices: list[int] | None = None):
        """Create static density snapshots using matplotlib."""
        if time_indices is None:
            time_indices = [0, len(self.solution.density_history) // 3, 2 * len(self.solution.density_history) // 3, -1]

        fig, axes = plt.subplots(1, len(time_indices), figsize=(4 * len(time_indices), 4))
        if len(time_indices) == 1:
            axes = [axes]

        for i, t_idx in enumerate(time_indices):
            m = self.solution.density_history[t_idx]
            im = axes[i].contourf(self.X, self.Y, m, levels=20, cmap="viridis")
            axes[i].set_title(f"t = {t_idx * 0.01:.2f}")
            axes[i].set_xlabel("x₁")
            axes[i].set_ylabel("x₂")
            plt.colorbar(im, ax=axes[i])

        if self.has_barriers:
            self._add_barriers_to_matplotlib_axes(axes)

        plt.tight_layout()
        plt.savefig("density_snapshots.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _create_metrics_plot_matplotlib(self, output_file: str):
        """Create static metrics plot using matplotlib."""
        time_grid = getattr(
            self.solution, "time_grid", np.linspace(0, self.problem.time_horizon, len(self.solution.density_history))
        )

        total_mass = [np.sum(m) for m in self.solution.density_history]
        peak_density = [np.max(m) for m in self.solution.density_history]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(time_grid, total_mass)
        axes[0].set_title("Total Mass Over Time")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Total Mass")

        axes[1].plot(time_grid, peak_density, color="red")
        axes[1].set_title("Peak Density Over Time")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Peak Density")

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

    def _add_barriers_to_plotly_figure(self, fig):
        """Add barrier shapes to plotly figure."""
        for barrier in self.problem.barriers:
            if hasattr(barrier, "center") and hasattr(barrier, "radius"):
                # Circular barrier
                fig.add_shape(
                    type="circle",
                    x0=barrier.center[0] - barrier.radius,
                    y0=barrier.center[1] - barrier.radius,
                    x1=barrier.center[0] + barrier.radius,
                    y1=barrier.center[1] + barrier.radius,
                    fillcolor="gray",
                    opacity=0.8,
                    line=dict(color="black", width=2),
                )

    def _add_barriers_to_matplotlib_axes(self, axes):
        """Add barrier shapes to matplotlib axes."""
        if not isinstance(axes, list):
            axes = [axes]

        for ax in axes:
            for barrier in self.problem.barriers:
                if hasattr(barrier, "center") and hasattr(barrier, "radius"):
                    # Circular barrier
                    circle = plt.Circle(barrier.center, barrier.radius, color="gray", alpha=0.8, zorder=10)
                    ax.add_patch(circle)
                elif hasattr(barrier, "start") and hasattr(barrier, "end"):
                    # Linear barrier
                    ax.plot(
                        [barrier.start[0], barrier.end[0]],
                        [barrier.start[1], barrier.end[1]],
                        "k-",
                        linewidth=8,
                        alpha=0.8,
                        zorder=10,
                    )


def create_visualization_suite(problem: AnisotropicCrowdDynamics, solution: Any, output_dir: str = "../results/"):
    """
    Create complete visualization suite for anisotropic experiment.

    Args:
        problem: Problem instance
        solution: Solution data
        output_dir: Output directory for visualizations
    """
    logger.info("Creating complete visualization suite")

    visualizer = AnisotropicVisualizer(problem, solution)

    # Create all visualizations
    visualizer.create_density_evolution_animation(f"{output_dir}density_evolution.html")
    visualizer.create_velocity_field_plot(output_file=f"{output_dir}velocity_field.png")
    visualizer.create_anisotropy_analysis_plot(f"{output_dir}anisotropy_analysis.png")

    if visualizer.has_barriers:
        visualizer.create_barrier_influence_analysis(f"{output_dir}barrier_influence.png")

    visualizer.create_evacuation_metrics_dashboard(f"{output_dir}metrics_dashboard.html")

    logger.info(f"Visualization suite created in {output_dir}")
    return visualizer


if __name__ == "__main__":
    print("Visualization tools module. Import and use with MFG solutions.")
    if PLOTLY_AVAILABLE:
        print("Plotly available: Interactive visualizations enabled.")
    else:
        print("Plotly not available: Using matplotlib only.")
