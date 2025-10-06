"""
Multi-dimensional visualization for 2D/3D MFG problems.

This module provides specialized visualization tools for high-dimensional Mean Field Games,
including 3D surface plots, slice visualizations, and interactive animations.

Features:
    - 3D surface plots for u(x,y,t) and m(x,y,t)
    - Contour and heatmap visualizations
    - Time evolution animations
    - Slice visualization across dimensions
    - Interactive Plotly-based plots
    - HTML export for presentations

Mathematical Context:
    For 2D MFG, we visualize:
        u(t,x,y): Value function over spatial domain
        m(t,x,y): Density distribution over spatial domain

    Common visualization tasks:
        - Surface: u(x,y) at fixed t
        - Evolution: u(x,y,t) animated over time
        - Slices: u(x₀,y) or u(x,y₀) cross-sections

References:
    - Plotly documentation for 3D visualizations
    - Hunter (2007): Matplotlib Tutorial
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.geometry.tensor_product_grid import TensorProductGrid


class MultiDimVisualizer:
    """
    Visualization manager for multi-dimensional MFG solutions.

    Provides unified interface for creating 3D visualizations of solutions
    on tensor product grids, supporting both Plotly (interactive) and
    Matplotlib (publication-quality) backends.

    Attributes:
        grid: Tensor product grid defining spatial domain
        backend: Visualization backend ('plotly' or 'matplotlib')
        colorscale: Default colorscale for plots

    Example:
        >>> from mfg_pde.geometry import TensorProductGrid
        >>> from mfg_pde.visualization import MultiDimVisualizer
        >>>
        >>> grid = TensorProductGrid(2, [(0,1), (0,1)], [51, 51])
        >>> viz = MultiDimVisualizer(grid, backend='plotly')
        >>>
        >>> # Create 3D surface plot
        >>> fig = viz.surface_plot(u, title='Value function u(x,y)')
        >>> viz.save(fig, 'value_function.html')
    """

    def __init__(
        self,
        grid: TensorProductGrid,
        backend: Literal["plotly", "matplotlib"] = "plotly",
        colorscale: str = "Viridis",
    ):
        """
        Initialize multi-dimensional visualizer.

        Args:
            grid: Tensor product grid for spatial discretization
            backend: Visualization backend (plotly/matplotlib)
            colorscale: Default colorscale (Viridis, Plasma, RdBu, etc.)

        Raises:
            ValueError: If grid dimension not in [2, 3]
            ImportError: If requested backend not available
        """
        if grid.dimension not in [2, 3]:
            raise ValueError(f"MultiDimVisualizer requires 2D or 3D grid, got {grid.dimension}D")

        self.grid = grid
        self.backend = backend
        self.colorscale = colorscale

        # Check backend availability
        if backend == "plotly":
            try:
                import plotly.graph_objects as go  # noqa: F401

                self.plotly_available = True
            except ImportError as e:
                raise ImportError("Plotly not available. Install with: pip install plotly") from e
        else:
            import matplotlib.pyplot as plt  # noqa: F401

            self.plotly_available = False

    def surface_plot(
        self,
        data: NDArray,
        title: str = "Surface Plot",
        xlabel: str = "x",
        ylabel: str = "y",
        zlabel: str = "u",
        time_index: int | None = None,
    ):
        """
        Create 3D surface plot of 2D data.

        Args:
            data: Solution array of shape (Nx, Ny) or (Nt, Nx, Ny)
            title: Plot title
            xlabel, ylabel, zlabel: Axis labels
            time_index: Time index if data has temporal dimension

        Returns:
            Plotly Figure or Matplotlib Figure

        Example:
            >>> # 2D value function at final time
            >>> u_final = u[-1, :, :]  # Shape: (Nx, Ny)
            >>> fig = viz.surface_plot(u_final, title='u(x,y) at T')
        """
        if self.grid.dimension != 2:
            raise ValueError("surface_plot requires 2D grid")

        # Extract 2D slice if data is 3D (temporal)
        if data.ndim == 3:
            if time_index is None:
                time_index = -1  # Default to final time
            plot_data = data[time_index, :, :]
            title = f"{title} (t={time_index})"
        elif data.ndim == 2:
            plot_data = data
        else:
            raise ValueError(f"Expected 2D or 3D data, got shape {data.shape}")

        X, Y = self.grid.meshgrid(indexing="ij")

        if self.backend == "plotly":
            return self._plotly_surface(X, Y, plot_data, title, xlabel, ylabel, zlabel)
        else:
            return self._matplotlib_surface(X, Y, plot_data, title, xlabel, ylabel, zlabel)

    def _plotly_surface(self, X, Y, Z, title, xlabel, ylabel, zlabel):
        """Create Plotly 3D surface plot."""
        import plotly.graph_objects as go

        fig = go.Figure(
            data=[
                go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    colorscale=self.colorscale,
                    colorbar={"title": zlabel},
                )
            ]
        )

        fig.update_layout(
            title=title,
            scene={
                "xaxis_title": xlabel,
                "yaxis_title": ylabel,
                "zaxis_title": zlabel,
                "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.3}},
            },
            width=800,
            height=600,
        )

        return fig

    def _matplotlib_surface(self, X, Y, Z, title, xlabel, ylabel, zlabel):
        """Create Matplotlib 3D surface plot."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(X, Y, Z, cmap=self.colorscale.lower(), linewidth=0, antialiased=True)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=zlabel)

        return fig

    def contour_plot(
        self,
        data: NDArray,
        title: str = "Contour Plot",
        xlabel: str = "x",
        ylabel: str = "y",
        time_index: int | None = None,
        levels: int = 20,
    ):
        """
        Create contour plot of 2D data.

        Args:
            data: Solution array of shape (Nx, Ny) or (Nt, Nx, Ny)
            title: Plot title
            xlabel, ylabel: Axis labels
            time_index: Time index if data has temporal dimension
            levels: Number of contour levels

        Returns:
            Plotly Figure or Matplotlib Figure
        """
        if self.grid.dimension != 2:
            raise ValueError("contour_plot requires 2D grid")

        # Extract 2D slice
        if data.ndim == 3:
            if time_index is None:
                time_index = -1
            plot_data = data[time_index, :, :]
            title = f"{title} (t={time_index})"
        else:
            plot_data = data

        X, Y = self.grid.meshgrid(indexing="ij")

        if self.backend == "plotly":
            return self._plotly_contour(X, Y, plot_data, title, xlabel, ylabel, levels)
        else:
            return self._matplotlib_contour(X, Y, plot_data, title, xlabel, ylabel, levels)

    def _plotly_contour(self, X, Y, Z, title, xlabel, ylabel, levels):
        """Create Plotly contour plot."""
        import plotly.graph_objects as go

        fig = go.Figure(
            data=go.Contour(
                x=X[0, :],
                y=Y[:, 0],
                z=Z,
                colorscale=self.colorscale,
                ncontours=levels,
                contours={"showlabels": True},
                colorbar={"title": "Value"},
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            width=700,
            height=600,
        )

        return fig

    def _matplotlib_contour(self, X, Y, Z, title, xlabel, ylabel, levels):
        """Create Matplotlib contour plot."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))

        contour = ax.contourf(X, Y, Z, levels=levels, cmap=self.colorscale.lower())
        ax.contour(X, Y, Z, levels=levels, colors="black", linewidths=0.5, alpha=0.4)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        fig.colorbar(contour, ax=ax, label="Value")

        return fig

    def heatmap(
        self,
        data: NDArray,
        title: str = "Heatmap",
        xlabel: str = "x",
        ylabel: str = "y",
        time_index: int | None = None,
    ):
        """
        Create heatmap visualization.

        Args:
            data: Solution array of shape (Nx, Ny) or (Nt, Nx, Ny)
            title: Plot title
            xlabel, ylabel: Axis labels
            time_index: Time index if data has temporal dimension

        Returns:
            Plotly Figure or Matplotlib Figure
        """
        if self.grid.dimension != 2:
            raise ValueError("heatmap requires 2D grid")

        # Extract 2D slice
        if data.ndim == 3:
            if time_index is None:
                time_index = -1
            plot_data = data[time_index, :, :]
            title = f"{title} (t={time_index})"
        else:
            plot_data = data

        if self.backend == "plotly":
            return self._plotly_heatmap(plot_data, title, xlabel, ylabel)
        else:
            return self._matplotlib_heatmap(plot_data, title, xlabel, ylabel)

    def _plotly_heatmap(self, Z, title, xlabel, ylabel):
        """Create Plotly heatmap."""
        import plotly.graph_objects as go

        fig = go.Figure(data=go.Heatmap(z=Z, colorscale=self.colorscale, colorbar={"title": "Value"}))

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            width=700,
            height=600,
        )

        return fig

    def _matplotlib_heatmap(self, Z, title, xlabel, ylabel):
        """Create Matplotlib heatmap."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(Z, cmap=self.colorscale.lower(), aspect="auto", origin="lower")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        fig.colorbar(im, ax=ax, label="Value")

        return fig

    def slice_plot(
        self,
        data: NDArray,
        slice_dim: int,
        slice_index: int,
        title: str = "Slice Plot",
        time_index: int | None = None,
    ):
        """
        Create visualization of data slice along specified dimension.

        Args:
            data: Solution array (2D or 3D spatial + optional time)
            slice_dim: Dimension to slice (0=x, 1=y, 2=z)
            slice_index: Index along slice dimension
            title: Plot title
            time_index: Time index if data has temporal dimension

        Returns:
            Plotly Figure or Matplotlib Figure

        Example:
            >>> # Slice at x = 0.5 (middle of domain)
            >>> fig = viz.slice_plot(u, slice_dim=0, slice_index=25)
        """
        # Extract spatial data
        if data.ndim == 4:  # (Nt, Nx, Ny, Nz)
            if time_index is None:
                time_index = -1
            spatial_data = data[time_index, ...]
        elif data.ndim == 3 and self.grid.dimension == 2:  # (Nt, Nx, Ny)
            if time_index is None:
                time_index = -1
            spatial_data = data[time_index, :, :]
        else:
            spatial_data = data

        # Extract slice
        if self.grid.dimension == 2:
            if slice_dim == 0:
                slice_data = spatial_data[slice_index, :]
                axis_label = "y"
                axis_coords = self.grid.coordinates[1]
            else:
                slice_data = spatial_data[:, slice_index]
                axis_label = "x"
                axis_coords = self.grid.coordinates[0]

            # 1D plot
            if self.backend == "plotly":
                import plotly.graph_objects as go

                fig = go.Figure(data=go.Scatter(x=axis_coords, y=slice_data, mode="lines"))
                fig.update_layout(title=title, xaxis_title=axis_label, yaxis_title="Value")
                return fig
            else:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(axis_coords, slice_data)
                ax.set_xlabel(axis_label)
                ax.set_ylabel("Value")
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                return fig

        elif self.grid.dimension == 3:
            # 3D slice produces 2D plot
            if slice_dim == 0:
                slice_data = spatial_data[slice_index, :, :]
                X, Y = np.meshgrid(self.grid.coordinates[1], self.grid.coordinates[2], indexing="ij")
                xlabel, ylabel = "y", "z"
            elif slice_dim == 1:
                slice_data = spatial_data[:, slice_index, :]
                X, Y = np.meshgrid(self.grid.coordinates[0], self.grid.coordinates[2], indexing="ij")
                xlabel, ylabel = "x", "z"
            else:
                slice_data = spatial_data[:, :, slice_index]
                X, Y = np.meshgrid(self.grid.coordinates[0], self.grid.coordinates[1], indexing="ij")
                xlabel, ylabel = "x", "y"

            return self._plotly_surface(X, Y, slice_data, title, xlabel, ylabel, "Value")

    def animation(
        self,
        data: NDArray,
        title: str = "Time Evolution",
        xlabel: str = "x",
        ylabel: str = "y",
        zlabel: str = "u",
        fps: int = 10,
    ):
        """
        Create animation of time evolution.

        Args:
            data: Temporal solution of shape (Nt, Nx, Ny)
            title: Animation title
            xlabel, ylabel, zlabel: Axis labels
            fps: Frames per second

        Returns:
            Plotly Figure with animation frames

        Note:
            Only available with Plotly backend
        """
        if self.backend != "plotly":
            raise ValueError("Animation requires Plotly backend")

        if self.grid.dimension != 2:
            raise ValueError("animation requires 2D grid")

        if data.ndim != 3:
            raise ValueError(f"Expected 3D data (Nt, Nx, Ny), got shape {data.shape}")

        import plotly.graph_objects as go

        X, Y = self.grid.meshgrid(indexing="ij")
        Nt = data.shape[0]

        # Create frames
        frames = []
        for t in range(Nt):
            frame = go.Frame(
                data=[
                    go.Surface(
                        x=X,
                        y=Y,
                        z=data[t, :, :],
                        colorscale=self.colorscale,
                        showscale=True,
                    )
                ],
                name=str(t),
            )
            frames.append(frame)

        # Initial frame
        fig = go.Figure(
            data=[
                go.Surface(
                    x=X,
                    y=Y,
                    z=data[0, :, :],
                    colorscale=self.colorscale,
                    colorbar={"title": zlabel},
                )
            ],
            frames=frames,
        )

        # Animation controls
        fig.update_layout(
            title=title,
            scene={
                "xaxis_title": xlabel,
                "yaxis_title": ylabel,
                "zaxis_title": zlabel,
                "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.3}},
            },
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 1000 // fps, "redraw": True},
                                    "fromcurrent": True,
                                    "mode": "immediate",
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                        },
                    ],
                }
            ],
            sliders=[
                {
                    "steps": [
                        {
                            "args": [
                                [str(t)],
                                {"frame": {"duration": 0}, "mode": "immediate"},
                            ],
                            "label": f"t={t}",
                            "method": "animate",
                        }
                        for t in range(Nt)
                    ],
                    "active": 0,
                    "y": 0,
                    "len": 0.9,
                    "x": 0.1,
                }
            ],
        )

        return fig

    def save(self, fig, filepath: str | Path):
        """
        Save figure to file.

        Args:
            fig: Figure object (Plotly or Matplotlib)
            filepath: Output file path (.html for Plotly, .png/.pdf for Matplotlib)
        """
        filepath = Path(filepath)

        if self.backend == "plotly":
            # Save as HTML
            fig.write_html(str(filepath))
        else:
            # Save as image
            fig.savefig(str(filepath), dpi=300, bbox_inches="tight")

    def show(self, fig):
        """
        Display figure interactively.

        Args:
            fig: Figure object to display
        """
        if self.backend == "plotly":
            fig.show()
        else:
            import matplotlib.pyplot as plt

            plt.show()
