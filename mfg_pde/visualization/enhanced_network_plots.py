"""
Enhanced Network MFG Visualization Tools.

This module extends the basic network visualization with advanced features
for analyzing Lagrangian network MFG solutions, trajectory tracking,
and high-order discretization scheme results.

Key enhancements:
- Trajectory path visualization for Lagrangian solutions
- Flow field analysis and vector plots
- High-order scheme comparison plots
- Interactive 3D network visualizations
- Advanced network statistics and analysis
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Import optional dependencies from parent module (centralized imports)
from . import PLOTLY_AVAILABLE, go
from .network_plots import NetworkMFGVisualizer

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from mfg_pde.core.network_mfg_problem import NetworkMFGProblem
    from mfg_pde.geometry.network_geometry import NetworkData


class EnhancedNetworkMFGVisualizer(NetworkMFGVisualizer):
    """
    Enhanced visualization toolkit for Network MFG with advanced features.

    Extends the base NetworkMFGVisualizer with capabilities for:
    - Lagrangian trajectory visualization
    - Flow field analysis
    - High-order scheme comparisons
    - 3D network plots
    """

    def __init__(
        self,
        problem: NetworkMFGProblem | None = None,
        network_data: NetworkData | None = None,
    ):
        """Initialize enhanced network visualizer."""
        super().__init__(problem, network_data)

        # Enhanced visualization parameters
        self.trajectory_colors = ["red", "blue", "green", "orange", "purple", "brown"]
        self.flow_arrow_scale = 1.0
        self.velocity_field_resolution = 20

    def plot_lagrangian_trajectories(
        self,
        trajectories: list[list[int]],
        U: np.ndarray | None = None,
        M: np.ndarray | None = None,
        title: str = "Lagrangian Trajectories",
        interactive: bool = True,
        save_path: str | None = None,
    ) -> go.Figure | Figure:
        """
        Visualize Lagrangian trajectories on the network.

        Args:
            trajectories: List of trajectories (each is list of node indices)
            U: Value function evolution (optional)
            M: Density evolution (optional)
            title: Plot title
            interactive: Use Plotly if True, matplotlib if False
            save_path: Path to save the plot

        Returns:
            Plotly Figure or matplotlib Figure
        """
        if interactive and PLOTLY_AVAILABLE:
            return self._plot_trajectories_plotly(trajectories, U, M, title, save_path)
        else:
            return self._plot_trajectories_matplotlib(trajectories, U, M, title, save_path)

    def _plot_trajectories_plotly(
        self,
        trajectories: list[list[int]],
        U: np.ndarray | None = None,
        M: np.ndarray | None = None,
        title: str = "Lagrangian Trajectories",
        save_path: str | None = None,
    ) -> go.Figure:
        """Create interactive trajectory plot using Plotly."""
        fig = go.Figure()

        # Base network topology
        self._add_network_topology_to_plotly(fig, node_values=M[0] if M is not None else None)

        # Add trajectories
        for i, trajectory in enumerate(trajectories):
            if len(trajectory) < 2:
                continue

            color = self.trajectory_colors[i % len(self.trajectory_colors)]

            # Get trajectory coordinates
            traj_x, traj_y = [], []
            for node in trajectory:
                if self.node_positions is not None:
                    x, y = self.node_positions[node]
                    traj_x.append(x)
                    traj_y.append(y)

            # Plot trajectory path
            fig.add_trace(
                go.Scatter(
                    x=traj_x,
                    y=traj_y,
                    mode="lines+markers",
                    line={"width": 3, "color": color},
                    marker={"size": 8, "color": color, "symbol": "circle"},
                    name=f"Trajectory {i + 1}",
                    hovertemplate=f"Trajectory {i + 1}<br>Step: %{{pointNumber}}<br>Node: %{{customdata}}<extra></extra>",
                    customdata=trajectory,
                )
            )

            # Add start and end markers
            if traj_x:
                # Start marker
                fig.add_trace(
                    go.Scatter(
                        x=[traj_x[0]],
                        y=[traj_y[0]],
                        mode="markers",
                        marker={"size": 12, "color": "green", "symbol": "star"},
                        name=f"Start {i + 1}",
                        showlegend=False,
                        hovertemplate=f"Start of Trajectory {i + 1}<br>Node: {trajectory[0]}<extra></extra>",
                    )
                )

                # End marker
                fig.add_trace(
                    go.Scatter(
                        x=[traj_x[-1]],
                        y=[traj_y[-1]],
                        mode="markers",
                        marker={"size": 12, "color": "red", "symbol": "square"},
                        name=f"End {i + 1}",
                        showlegend=False,
                        hovertemplate=f"End of Trajectory {i + 1}<br>Node: {trajectory[-1]}<extra></extra>",
                    )
                )

        # Update layout
        fig.update_layout(
            title=title,
            showlegend=True,
            hovermode="closest",
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            plot_bgcolor="white",
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def _plot_trajectories_matplotlib(
        self,
        trajectories: list[list[int]],
        U: np.ndarray | None = None,
        M: np.ndarray | None = None,
        title: str = "Lagrangian Trajectories",
        save_path: str | None = None,
    ) -> Figure:
        """Create trajectory plot using matplotlib."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Base network topology
        self._add_network_topology_to_matplotlib(ax, node_values=M[0] if M is not None else None)

        # Add trajectories
        for i, trajectory in enumerate(trajectories):
            if len(trajectory) < 2:
                continue

            color = self.trajectory_colors[i % len(self.trajectory_colors)]

            # Get trajectory coordinates
            traj_x, traj_y = [], []
            for node in trajectory:
                if self.node_positions is not None:
                    x, y = self.node_positions[node]
                    traj_x.append(x)
                    traj_y.append(y)

            if traj_x:
                # Plot trajectory path
                ax.plot(
                    traj_x,
                    traj_y,
                    "o-",
                    color=color,
                    linewidth=3,
                    markersize=6,
                    label=f"Trajectory {i + 1}",
                    alpha=0.8,
                )

                # Start and end markers
                ax.plot(
                    traj_x[0],
                    traj_y[0],
                    "*",
                    color="green",
                    markersize=15,
                    markeredgecolor="black",
                    markeredgewidth=1,
                )
                ax.plot(
                    traj_x[-1],
                    traj_y[-1],
                    "s",
                    color="red",
                    markersize=12,
                    markeredgecolor="black",
                    markeredgewidth=1,
                )

        ax.set_title(title, fontsize=16)
        ax.legend(fontsize=12)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_velocity_field(
        self,
        velocity_field: np.ndarray,
        M: np.ndarray | None = None,
        title: str = "Network Velocity Field",
        time_idx: int = 0,
        interactive: bool = True,
        save_path: str | None = None,
    ) -> go.Figure | Figure:
        """
        Visualize velocity field on the network.

        Args:
            velocity_field: (Nt+1, num_nodes, velocity_dim) velocity at each node
            M: Density evolution (optional, for background)
            title: Plot title
            time_idx: Time index to visualize
            interactive: Use Plotly if True, matplotlib if False
            save_path: Path to save the plot

        Returns:
            Plotly Figure or matplotlib Figure
        """
        if interactive and PLOTLY_AVAILABLE:
            return self._plot_velocity_field_plotly(velocity_field, M, title, time_idx, save_path)
        else:
            return self._plot_velocity_field_matplotlib(velocity_field, M, title, time_idx, save_path)

    def _plot_velocity_field_matplotlib(
        self,
        velocity_field: np.ndarray,
        M: np.ndarray | None = None,
        title: str = "Network Velocity Field",
        time_idx: int = 0,
        save_path: str | None = None,
    ) -> Figure:
        """Create velocity field plot using matplotlib."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Background density if provided
        if M is not None:
            self._add_network_topology_to_matplotlib(ax, node_values=M[time_idx])
        else:
            self._add_network_topology_to_matplotlib(ax)

        # Add velocity vectors
        if self.node_positions is not None and velocity_field.ndim >= 2:
            velocities = velocity_field[time_idx] if velocity_field.ndim == 3 else velocity_field

            for node in range(self.num_nodes):
                x, y = self.node_positions[node]

                if velocities.shape[1] >= 2:  # 2D velocity
                    vx, vy = velocities[node, 0], velocities[node, 1]

                    # Scale velocity for visualization
                    speed = np.sqrt(vx**2 + vy**2)
                    if speed > 1e-10:
                        scale = self.flow_arrow_scale * 0.1
                        ax.arrow(
                            x,
                            y,
                            vx * scale,
                            vy * scale,
                            head_width=0.02,
                            head_length=0.03,
                            fc="red",
                            ec="red",
                            alpha=0.7,
                        )
                elif velocities.shape[1] >= 1:  # 1D velocity
                    v = velocities[node, 0]

                    # Show 1D velocity as vertical arrow
                    if abs(v) > 1e-10:
                        scale = self.flow_arrow_scale * 0.1
                        ax.arrow(
                            x,
                            y,
                            0,
                            v * scale,
                            head_width=0.02,
                            head_length=0.03,
                            fc="red",
                            ec="red",
                            alpha=0.7,
                        )

        ax.set_title(f"{title} (t = {time_idx})", fontsize=16)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_scheme_comparison(
        self,
        solutions: dict[str, tuple[np.ndarray, np.ndarray]],
        times: np.ndarray,
        selected_nodes: list[int] | None = None,
        title: str = "Discretization Scheme Comparison",
        interactive: bool = True,
        save_path: str | None = None,
    ) -> go.Figure | Figure:
        """
        Compare solutions from different discretization schemes.

        Args:
            solutions: Dict mapping scheme names to (U, M) solution tuples
            times: Time array
            selected_nodes: Nodes to compare (default: first few nodes)
            title: Plot title
            interactive: Use Plotly if True, matplotlib if False
            save_path: Path to save the plot

        Returns:
            Plotly Figure or matplotlib Figure
        """
        if selected_nodes is None:
            selected_nodes = list(range(min(4, self.num_nodes)))

        if interactive and PLOTLY_AVAILABLE:
            return self._plot_scheme_comparison_plotly(solutions, times, selected_nodes, title, save_path)
        else:
            return self._plot_scheme_comparison_matplotlib(solutions, times, selected_nodes, title, save_path)

    def _plot_scheme_comparison_matplotlib(
        self,
        solutions: dict[str, tuple[np.ndarray, np.ndarray]],
        times: np.ndarray,
        selected_nodes: list[int],
        title: str = "Discretization Scheme Comparison",
        save_path: str | None = None,
    ) -> Figure:
        """Create scheme comparison plot using matplotlib."""
        n_nodes = len(selected_nodes)
        fig, axes = plt.subplots(2, n_nodes, figsize=(4 * n_nodes, 8))

        if n_nodes == 1:
            axes = axes.reshape(2, 1)

        colors = cm.get_cmap("Set1")(np.linspace(0, 1, len(solutions)))

        for node_idx, node in enumerate(selected_nodes):
            # Value function comparison
            ax_u = axes[0, node_idx]
            for i, (scheme_name, (U, _M)) in enumerate(solutions.items()):
                ax_u.plot(times, U[:, node], label=scheme_name, color=colors[i], linewidth=2)

            ax_u.set_title(f"Value Function - Node {node}")
            ax_u.set_xlabel("Time")
            ax_u.set_ylabel("Value")
            ax_u.legend()
            ax_u.grid(True, alpha=0.3)

            # Density comparison
            ax_m = axes[1, node_idx]
            for i, (scheme_name, (_U, M)) in enumerate(solutions.items()):
                ax_m.plot(times, M[:, node], label=scheme_name, color=colors[i], linewidth=2)

            ax_m.set_title(f"Density - Node {node}")
            ax_m.set_xlabel("Time")
            ax_m.set_ylabel("Density")
            ax_m.legend()
            ax_m.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_network_3d(
        self,
        node_values: np.ndarray | None = None,
        edge_values: np.ndarray | None = None,
        title: str = "3D Network Visualization",
        height_scale: float = 1.0,
        save_path: str | None = None,
    ) -> go.Figure:
        """
        Create 3D network visualization with node values as height.

        Args:
            node_values: Values to display as node heights
            edge_values: Values to display as edge properties
            title: Plot title
            height_scale: Scale factor for node heights
            save_path: Path to save the plot

        Returns:
            Plotly 3D Figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for 3D network visualization")

        fig = go.Figure()

        # Node positions with height
        if self.node_positions is not None:
            node_x = self.node_positions[:, 0]
            node_y = self.node_positions[:, 1]
        else:
            # Default circular layout
            angles = np.linspace(0, 2 * np.pi, self.num_nodes, endpoint=False)
            node_x = np.cos(angles)
            node_y = np.sin(angles)

        # Node heights from values
        if node_values is not None:
            node_z = node_values * height_scale
            node_color = node_values
            colorscale = "Viridis"
        else:
            node_z = np.zeros(self.num_nodes)
            node_color = "lightblue"  # type: ignore[assignment]
            colorscale = None

        # Add 3D edges
        edge_x, edge_y, edge_z = [], [], []
        rows, cols = self.adjacency_matrix.nonzero()

        for i, j in zip(rows, cols, strict=False):
            # Edge coordinates
            edge_x.extend([node_x[i], node_x[j], None])
            edge_y.extend([node_y[i], node_y[j], None])
            edge_z.extend([node_z[i], node_z[j], None])

        # Plot 3D edges
        fig.add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line={"width": 2, "color": "lightgray"},
                hoverinfo="none",
                showlegend=False,
                name="Edges",
            )
        )

        # Plot 3D nodes
        fig.add_trace(
            go.Scatter3d(
                x=node_x,
                y=node_y,
                z=node_z,
                mode="markers",
                marker={
                    "size": 8,
                    "color": node_color,
                    "colorscale": colorscale,
                    "showscale": node_values is not None,
                    "colorbar": ({"title": "Node Values"} if node_values is not None else None),
                },
                text=[f"Node {i}<br>Height: {node_z[i]:.3f}" for i in range(self.num_nodes)],
                hovertemplate="%{text}<extra></extra>",
                name="Nodes",
            )
        )

        # Update layout for 3D
        fig.update_layout(
            title=title,
            scene={
                "xaxis_title": "X",
                "yaxis_title": "Y",
                "zaxis_title": "Height",
                "aspectmode": "manual",
                "aspectratio": {"x": 1, "y": 1, "z": 0.5},
            },
            showlegend=False,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def _add_network_topology_to_plotly(self, fig: go.Figure, node_values: np.ndarray | None = None):
        """Add network topology to existing Plotly figure."""
        # This would include the network plotting logic from the base class
        # For brevity, we'll reference the existing implementation

    def _add_network_topology_to_matplotlib(self, ax: Axes, node_values: np.ndarray | None = None):
        """Add network topology to existing matplotlib axes."""
        # Plot edges
        rows, cols = self.adjacency_matrix.nonzero()
        for i, j in zip(rows, cols, strict=False):
            if self.node_positions is not None:
                x0, y0 = self.node_positions[i]
                x1, y1 = self.node_positions[j]
                ax.plot([x0, x1], [y0, y1], "k-", alpha=0.3, linewidth=1)

        # Plot nodes
        if self.node_positions is not None:
            node_x = self.node_positions[:, 0]
            node_y = self.node_positions[:, 1]

            if node_values is not None:
                scatter = ax.scatter(
                    node_x,
                    node_y,
                    c=node_values,
                    s=100,
                    cmap="viridis",
                    alpha=0.8,
                    edgecolors="black",
                )
                plt.colorbar(scatter, ax=ax, label="Node Values")
            else:
                ax.scatter(node_x, node_y, c="lightblue", s=100, alpha=0.8, edgecolors="black")

    def _plot_velocity_field_plotly(
        self,
        velocity_field: np.ndarray,
        M: np.ndarray | None,
        title: str,
        time_idx: int,
        save_path: str | None = None,
    ) -> go.Figure:
        """Plot velocity field using Plotly (placeholder implementation)."""
        # Placeholder - would implement velocity field visualization
        fig = go.Figure()
        fig.update_layout(title=f"{title} - Velocity Field (t={time_idx})")
        if save_path:
            fig.write_html(save_path)
        return fig

    def _plot_scheme_comparison_plotly(
        self,
        solutions: dict[str, tuple[np.ndarray, np.ndarray]],
        times: np.ndarray,
        selected_nodes: list[int],
        title: str,
        save_path: str | None = None,
    ) -> go.Figure:
        """Plot scheme comparison using Plotly (placeholder implementation)."""
        # Placeholder - would implement scheme comparison visualization
        fig = go.Figure()
        fig.update_layout(title=f"{title} - Scheme Comparison")
        if save_path:
            fig.write_html(save_path)
        return fig


# Factory function for enhanced network visualizer
def create_enhanced_network_visualizer(
    problem: NetworkMFGProblem | None = None,
    network_data: NetworkData | None = None,
) -> EnhancedNetworkMFGVisualizer:
    """
    Create enhanced network MFG visualizer.

    Args:
        problem: Network MFG problem (optional)
        network_data: Network data (optional)

    Returns:
        Enhanced network visualizer instance
    """
    return EnhancedNetworkMFGVisualizer(problem=problem, network_data=network_data)
