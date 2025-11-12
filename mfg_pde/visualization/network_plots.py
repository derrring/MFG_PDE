"""
Network MFG Visualization Tools.

This module provides comprehensive visualization capabilities for Mean Field Games
on network structures, including interactive network plots, flow animations,
and density evolution visualizations.

Key features:
- Interactive network topology visualization
- Node density and value evolution plots
- Edge flow visualization and animation
- Network statistics and analysis plots
- Integration with Plotly, NetworkX, and matplotlib
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Import optional dependencies from parent module (centralized imports)
from . import NETWORKX_AVAILABLE, PLOTLY_AVAILABLE, go, make_subplots, nx, px

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from mfg_pde.extensions.topology import NetworkMFGProblem
    from mfg_pde.geometry.network_geometry import NetworkData


class NetworkMFGVisualizer:
    """
    Comprehensive visualization toolkit for Network MFG problems and solutions.

    Provides static plots, interactive visualizations, and animations
    for network topology, density evolution, and flow dynamics.
    """

    def __init__(
        self,
        problem: NetworkMFGProblem | None = None,
        network_data: NetworkData | None = None,
    ):
        """
        Initialize network MFG visualizer.

        Args:
            problem: Network MFG problem instance (optional)
            network_data: Network data structure (optional)
        """
        self.problem = problem
        self.network_data = network_data or (problem.network_data if problem else None)

        if self.network_data is None:
            raise ValueError("Either problem or network_data must be provided")

        # Network properties
        self.num_nodes = self.network_data.num_nodes
        self.num_edges = self.network_data.num_edges
        self.adjacency_matrix = self.network_data.adjacency_matrix
        self.node_positions = self.network_data.node_positions

        # Default visualization parameters
        self.default_node_size = 300
        self.default_edge_width = 2
        self.default_colorscale = "viridis"

    def plot_network_topology(
        self,
        node_values: np.ndarray | None = None,
        edge_values: np.ndarray | None = None,
        title: str = "Network Topology",
        node_size_scale: float = 1.0,
        edge_width_scale: float = 1.0,
        interactive: bool = True,
        save_path: str | None = None,
    ) -> Any:
        """
        Plot network topology with optional node and edge coloring.

        Args:
            node_values: Values to color nodes (optional)
            edge_values: Values to color edges (optional)
            title: Plot title
            node_size_scale: Scale factor for node sizes
            edge_width_scale: Scale factor for edge widths
            interactive: Use interactive plotting if available
            save_path: Path to save plot (optional)

        Returns:
            Plot object (matplotlib figure or plotly figure)
        """
        if interactive and PLOTLY_AVAILABLE:
            return self._plot_network_plotly(
                node_values,
                edge_values,
                title,
                node_size_scale,
                edge_width_scale,
                save_path,
            )
        else:
            return self._plot_network_matplotlib(
                node_values,
                edge_values,
                title,
                node_size_scale,
                edge_width_scale,
                save_path,
            )

    def _plot_network_plotly(
        self,
        node_values: np.ndarray | None = None,
        edge_values: np.ndarray | None = None,
        title: str = "Network Topology",
        node_size_scale: float = 1.0,
        edge_width_scale: float = 1.0,
        save_path: str | None = None,
    ) -> go.Figure:
        """Create interactive network plot using Plotly."""
        fig = go.Figure()

        # Extract edge coordinates
        edge_x, edge_y = [], []
        rows, cols = self.adjacency_matrix.nonzero()

        for i, j in zip(rows, cols, strict=False):
            if self.node_positions is not None:
                x0, y0 = self.node_positions[i]
                x1, y1 = self.node_positions[j]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

        # Plot edges
        edge_color = "lightgray"
        edge_width = self.default_edge_width * edge_width_scale

        if edge_values is not None:
            # Color edges by values (simplified - would need proper edge indexing)
            edge_color = "blue"

        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"width": edge_width, "color": edge_color},
                hoverinfo="none",
                showlegend=False,
                name="Edges",
            )
        )

        # Plot nodes
        if self.node_positions is not None:
            node_x = self.node_positions[:, 0]
            node_y = self.node_positions[:, 1]
        else:
            # Default circular layout
            angles = np.linspace(0, 2 * np.pi, self.num_nodes, endpoint=False)
            node_x = np.cos(angles)
            node_y = np.sin(angles)

        # Node colors and sizes
        node_color = "lightblue"
        node_size = self.default_node_size * node_size_scale

        if node_values is not None:
            node_color = node_values  # type: ignore[assignment]
            colorscale = self.default_colorscale
        else:
            colorscale = None

        # Node hover text
        if self.network_data is not None:
            node_text = [f"Node {i}<br>Degree: {self.network_data.get_node_degree(i)}" for i in range(self.num_nodes)]
        else:
            node_text = [f"Node {i}" for i in range(self.num_nodes)]

        if node_values is not None:
            node_text = [f"{text}<br>Value: {node_values[i]:.3f}" for i, text in enumerate(node_text)]

        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker={
                    "size": [node_size / 10] * self.num_nodes,  # Plotly uses different scaling
                    "color": node_color,
                    "colorscale": colorscale,
                    "showscale": node_values is not None,
                    "colorbar": ({"title": "Node Values"} if node_values is not None else None),
                    "line": {"width": 2, "color": "black"},
                },
                text=[str(i) for i in range(self.num_nodes)],
                textposition="middle center",
                hovertext=node_text,
                hoverinfo="text",
                name="Nodes",
            )
        )

        # Layout
        fig.update_layout(
            title=title,
            showlegend=False,
            hovermode="closest",
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            plot_bgcolor="white",
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def _plot_network_matplotlib(
        self,
        node_values: np.ndarray | None = None,
        edge_values: np.ndarray | None = None,
        title: str = "Network Topology",
        node_size_scale: float = 1.0,
        edge_width_scale: float = 1.0,
        save_path: str | None = None,
    ) -> Figure:
        """Create network plot using matplotlib."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Use NetworkX for layout if available
        if NETWORKX_AVAILABLE:
            G = nx.from_scipy_sparse_array(self.adjacency_matrix)

            if self.node_positions is not None:
                pos = {i: self.node_positions[i] for i in range(self.num_nodes)}
            else:
                pos = nx.spring_layout(G)

            # Draw edges
            edge_width = self.default_edge_width * edge_width_scale
            nx.draw_networkx_edges(G, pos, ax=ax, width=edge_width, alpha=0.6, edge_color="gray")

            # Draw nodes
            node_size = self.default_node_size * node_size_scale

            if node_values is not None:
                nodes = nx.draw_networkx_nodes(
                    G,
                    pos,
                    ax=ax,
                    node_color=node_values,
                    node_size=node_size,
                    cmap=self.default_colorscale,
                    vmin=np.min(node_values),
                    vmax=np.max(node_values),
                )
                plt.colorbar(nodes, ax=ax, label="Node Values")
            else:
                nx.draw_networkx_nodes(G, pos, ax=ax, node_color="lightblue", node_size=node_size)

            # Draw node labels
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

        else:
            # Fallback: simple matplotlib plotting
            ax.text(
                0.5,
                0.5,
                "NetworkX not available\nCannot plot network",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_title(title)
        ax.axis("off")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_density_evolution(
        self,
        M: np.ndarray,
        times: np.ndarray | None = None,
        selected_nodes: list[int] | None = None,
        title: str = "Density Evolution",
        interactive: bool = True,
        save_path: str | None = None,
    ) -> Any:
        """
        Plot density evolution over time for selected nodes.

        Args:
            M: (Nt+1, num_nodes) density evolution
            times: Time points (optional)
            selected_nodes: Nodes to plot (optional, defaults to all)
            title: Plot title
            interactive: Use interactive plotting
            save_path: Path to save plot

        Returns:
            Plot object
        """
        Nt, num_nodes = M.shape

        if times is None:
            times = np.linspace(0, 1, Nt)

        if selected_nodes is None:
            # Select nodes with highest average density
            avg_density = np.mean(M, axis=0)
            selected_nodes = np.argsort(avg_density)[-min(10, num_nodes) :].tolist()

        # At this point selected_nodes is guaranteed to be a list
        assert selected_nodes is not None

        if interactive and PLOTLY_AVAILABLE:
            return self._plot_density_evolution_plotly(M, times, selected_nodes, title, save_path)
        else:
            return self._plot_density_evolution_matplotlib(M, times, selected_nodes, title, save_path)

    def _plot_density_evolution_plotly(
        self,
        M: np.ndarray,
        times: np.ndarray,
        selected_nodes: list[int],
        title: str,
        save_path: str | None = None,
    ) -> go.Figure:
        """Plot density evolution using Plotly."""
        fig = go.Figure()

        colors = px.colors.qualitative.Set1

        for i, node in enumerate(selected_nodes):
            color = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=M[:, node],
                    mode="lines+markers",
                    name=f"Node {node}",
                    line={"color": color, "width": 2},
                    marker={"size": 4},
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Density",
            hovermode="x unified",
            showlegend=True,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def _plot_density_evolution_matplotlib(
        self,
        M: np.ndarray,
        times: np.ndarray,
        selected_nodes: list[int],
        title: str,
        save_path: str | None = None,
    ) -> Figure:
        """Plot density evolution using matplotlib."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for node in selected_nodes:
            ax.plot(times, M[:, node], label=f"Node {node}", marker="o", markersize=3)

        ax.set_xlabel("Time")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_value_function_evolution(
        self,
        U: np.ndarray,
        times: np.ndarray | None = None,
        selected_nodes: list[int] | None = None,
        title: str = "Value Function Evolution",
        interactive: bool = True,
        save_path: str | None = None,
    ) -> Any:
        """
        Plot value function evolution over time.

        Similar to density evolution but for value functions.
        """
        # Reuse density evolution plotting with different title
        return self.plot_density_evolution(U, times, selected_nodes, title, interactive, save_path)

    def create_flow_animation(
        self,
        U: np.ndarray,
        M: np.ndarray,
        times: np.ndarray | None = None,
        interval: int = 200,
        save_path: str | None = None,
    ) -> FuncAnimation:
        """
        Create animation of density and flow evolution on network.

        Args:
            U: (Nt+1, num_nodes) value function evolution
            M: (Nt+1, num_nodes) density evolution
            times: Time points
            interval: Animation interval in milliseconds
            save_path: Path to save animation

        Returns:
            Matplotlib animation object
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX required for flow animation")

        Nt, _num_nodes = M.shape

        if times is None:
            times = np.linspace(0, 1, Nt)

        # Create network graph
        G = nx.from_scipy_sparse_array(self.adjacency_matrix)

        if self.node_positions is not None:
            pos = {i: self.node_positions[i] for i in range(self.num_nodes)}
        else:
            pos = nx.spring_layout(G)

        # Setup figure
        fig, ax = plt.subplots(figsize=(12, 8))

        def animate(frame):
            ax.clear()

            # Current time step
            t_idx = frame
            current_time = times[t_idx]
            current_density = M[t_idx, :]
            current_value = U[t_idx, :]

            # Draw edges
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color="gray")

            # Draw nodes with density-based sizing and value-based coloring
            node_sizes = 200 + 1000 * current_density / np.max(current_density)

            nodes = nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                node_color=current_value,
                node_size=node_sizes,
                cmap="RdYlBu",
                vmin=np.min(U),
                vmax=np.max(U),
                alpha=0.8,
            )

            # Draw node labels
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

            ax.set_title(f"Network MFG Evolution - Time: {current_time:.3f}")
            ax.axis("off")

            # Add colorbar for value function
            if frame == 0:
                cbar = plt.colorbar(nodes, ax=ax, shrink=0.8)
                cbar.set_label("Value Function")

            return [nodes] if nodes else []

        # Create animation
        anim = FuncAnimation(fig, animate, frames=Nt, interval=interval, repeat=True)

        if save_path:
            anim.save(save_path, writer="pillow" if save_path.endswith(".gif") else "ffmpeg")

        return anim

    def plot_network_statistics_dashboard(
        self,
        convergence_info: dict[str, Any] | None = None,
        save_path: str | None = None,
    ) -> Any:
        """
        Create comprehensive dashboard of network statistics and analysis.

        Args:
            convergence_info: Convergence information from solver
            save_path: Path to save dashboard

        Returns:
            Dashboard plot object
        """
        if PLOTLY_AVAILABLE:
            return self._create_plotly_dashboard(convergence_info, save_path)
        else:
            return self._create_matplotlib_dashboard(convergence_info, save_path)

    def _create_plotly_dashboard(
        self,
        convergence_info: dict[str, Any] | None = None,
        save_path: str | None = None,
    ) -> go.Figure:
        """Create interactive dashboard using Plotly."""
        from mfg_pde.geometry.network_geometry import compute_network_statistics

        # Network statistics
        stats = compute_network_statistics(self.network_data)

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Network Properties",
                "Degree Distribution",
                "Convergence History",
                "Mass Conservation",
            ],
            specs=[
                [{"type": "table"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "scatter"}],
            ],
        )

        # Network properties table
        properties = [
            ["Nodes", stats["num_nodes"]],
            ["Edges", stats["num_edges"]],
            ["Density", f"{stats['density']:.3f}"],
            ["Average Degree", f"{stats['average_degree']:.2f}"],
            ["Max Degree", stats["max_degree"]],
            ["Connected", "Yes" if stats["is_connected"] else "No"],
            ["Clustering Coeff.", f"{stats['clustering_coefficient']:.3f}"],
        ]

        fig.add_trace(
            go.Table(
                header={"values": ["Property", "Value"]},
                cells={"values": list(zip(*properties, strict=False))},
            ),
            row=1,
            col=1,
        )

        # Degree distribution
        degrees = np.array(self.adjacency_matrix.sum(axis=1)).flatten()
        fig.add_trace(
            go.Histogram(
                x=degrees,
                nbinsx=min(20, len(np.unique(degrees))),
                name="Degree Distribution",
            ),
            row=1,
            col=2,
        )

        # Convergence history
        if convergence_info and "convergence_history" in convergence_info:
            history = convergence_info["convergence_history"]
            iterations = [h["iteration"] for h in history]
            u_errors = [h["u_error"] for h in history]
            m_errors = [h["m_error"] for h in history]

            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=u_errors,
                    mode="lines+markers",
                    name="U Error",
                    yaxis="y3",
                ),
                row=2,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=m_errors,
                    mode="lines+markers",
                    name="M Error",
                    yaxis="y3",
                ),
                row=2,
                col=1,
            )

            # Mass conservation
            if "mass_conservation" in history[0]:
                times = list(range(len(history[0]["mass_conservation"])))
                for _i, h in enumerate(history[::5]):  # Sample every 5th iteration
                    fig.add_trace(
                        go.Scatter(
                            x=times,
                            y=h["mass_conservation"],
                            mode="lines",
                            name=f"Iter {h['iteration']}",
                            opacity=0.7,
                        ),
                        row=2,
                        col=2,
                    )

        fig.update_layout(title="Network MFG Analysis Dashboard", height=800, showlegend=True)

        # Update y-axes to log scale for convergence
        fig.update_yaxes(type="log", row=2, col=1)

        if save_path:
            fig.write_html(save_path)

        return fig

    def _create_matplotlib_dashboard(
        self,
        convergence_info: dict[str, Any] | None = None,
        save_path: str | None = None,
    ) -> Figure:
        """Create dashboard using matplotlib."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Network MFG Analysis Dashboard", fontsize=16)

        # Network properties (top-left)
        ax = axes[0, 0]
        from mfg_pde.geometry.network_geometry import compute_network_statistics

        stats = compute_network_statistics(self.network_data)

        properties_text = f"""
        Nodes: {stats["num_nodes"]}
        Edges: {stats["num_edges"]}
        Density: {stats["density"]:.3f}
        Avg Degree: {stats["average_degree"]:.2f}
        Max Degree: {stats["max_degree"]}
        Connected: {"Yes" if stats["is_connected"] else "No"}
        Clustering: {stats["clustering_coefficient"]:.3f}
        """

        ax.text(
            0.1,
            0.5,
            properties_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="center",
        )
        ax.set_title("Network Properties")
        ax.axis("off")

        # Degree distribution (top-right)
        ax = axes[0, 1]
        degrees = np.array(self.adjacency_matrix.sum(axis=1)).flatten()
        ax.hist(degrees, bins=min(20, len(np.unique(degrees))), alpha=0.7, edgecolor="black")
        ax.set_xlabel("Degree")
        ax.set_ylabel("Count")
        ax.set_title("Degree Distribution")
        ax.grid(True, alpha=0.3)

        # Convergence history (bottom-left)
        ax = axes[1, 0]
        if convergence_info and "convergence_history" in convergence_info:
            history = convergence_info["convergence_history"]
            iterations = [h["iteration"] for h in history]
            u_errors = [h["u_error"] for h in history]
            m_errors = [h["m_error"] for h in history]

            ax.semilogy(iterations, u_errors, "o-", label="U Error", markersize=4)
            ax.semilogy(iterations, m_errors, "s-", label="M Error", markersize=4)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Error")
            ax.set_title("Convergence History")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5,
                0.5,
                "No convergence data available",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title("Convergence History")

        # Mass conservation (bottom-right)
        ax = axes[1, 1]
        if (
            convergence_info
            and "convergence_history" in convergence_info
            and "mass_conservation" in convergence_info["convergence_history"][0]
        ):
            history = convergence_info["convergence_history"]
            times = list(range(len(history[0]["mass_conservation"])))

            for _i, h in enumerate(history[:: max(1, len(history) // 5)]):
                ax.plot(
                    times,
                    h["mass_conservation"],
                    alpha=0.7,
                    label=f"Iter {h['iteration']}",
                )

            ax.set_xlabel("Time Step")
            ax.set_ylabel("Total Mass")
            ax.set_title("Mass Conservation")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5,
                0.5,
                "No mass conservation data available",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title("Mass Conservation")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


# Factory function for creating network visualizers
def create_network_visualizer(
    problem: NetworkMFGProblem | None = None,
    network_data: NetworkData | None = None,
) -> NetworkMFGVisualizer:
    """
    Create network MFG visualizer.

    Args:
        problem: Network MFG problem (optional)
        network_data: Network data structure (optional)

    Returns:
        Configured network visualizer
    """
    return NetworkMFGVisualizer(problem=problem, network_data=network_data)
