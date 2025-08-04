"""
Complete MFG Analytics and Visualization System.

This module provides a comprehensive analytics platform for Mean Field Games,
integrating Polars data manipulation with advanced Plotly/Bokeh visualizations
for professional research and analysis workflows.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..utils.integration import trapezoid

# Import MFG components
try:
    from .interactive_plots import (
        BOKEH_AVAILABLE,
        PLOTLY_AVAILABLE,
        MFGVisualizationManager,
        create_visualization_manager,
    )

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    create_visualization_manager = None
    MFGVisualizationManager = None

try:
    from ..utils.polars_integration import (
        POLARS_AVAILABLE,
        MFGDataFrame,
        create_data_exporter,
        create_parameter_sweep_analyzer,
        create_time_series_analyzer,
    )
except ImportError:
    POLARS_AVAILABLE = False
    create_parameter_sweep_analyzer = None
    create_time_series_analyzer = None
    create_data_exporter = None
    MFGDataFrame = None

logger = logging.getLogger(__name__)


class MFGAnalyticsEngine:
    """
    Comprehensive analytics engine for MFG research combining data analysis and visualization.

    Features:
    - High-performance data processing with Polars
    - Interactive 2D/3D visualizations with Plotly/Bokeh
    - Parameter sweep analysis and optimization
    - Convergence monitoring and analysis
    - Publication-quality exports
    - Integrated research workflows
    """

    def __init__(self, prefer_plotly: bool = True, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize MFG Analytics Engine.

        Args:
            prefer_plotly: Whether to prefer Plotly over Bokeh
            output_dir: Directory for saving outputs
        """
        self.prefer_plotly = prefer_plotly
        self.output_dir = Path(output_dir) if output_dir else Path("mfg_analytics_results")
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.viz_manager = None
        self.sweep_analyzer = None
        self.ts_analyzer = None
        self.data_exporter = None

        self._initialize_components()

    def _initialize_components(self):
        """Initialize analytics components."""
        # Visualization manager
        if VISUALIZATION_AVAILABLE:
            try:
                self.viz_manager = create_visualization_manager(self.prefer_plotly)
                logger.info("Visualization manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize visualization: {e}")

        # Data analysis components
        if POLARS_AVAILABLE:
            try:
                self.sweep_analyzer = create_parameter_sweep_analyzer()
                self.ts_analyzer = create_time_series_analyzer()
                self.data_exporter = create_data_exporter()
                logger.info("Polars analytics components initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize data analytics: {e}")

    def get_capabilities(self) -> Dict[str, bool]:
        """Get available analytics capabilities."""
        return {
            "visualization": self.viz_manager is not None,
            "data_analysis": POLARS_AVAILABLE,
            "plotly_3d": PLOTLY_AVAILABLE,
            "bokeh_interactive": BOKEH_AVAILABLE,
            "parameter_sweeps": self.sweep_analyzer is not None,
            "convergence_analysis": self.ts_analyzer is not None,
            "data_export": self.data_exporter is not None,
        }

    def analyze_mfg_solution(
        self,
        x_grid: np.ndarray,
        time_grid: np.ndarray,
        density_history: np.ndarray,
        value_history: np.ndarray,
        convergence_data: Optional[List[Dict]] = None,
        title: str = "MFG Solution Analysis",
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of MFG solution with visualizations.

        Args:
            x_grid: Spatial grid points
            time_grid: Time grid points
            density_history: Density evolution [time, space]
            value_history: Value function evolution [time, space]
            convergence_data: Optional convergence history
            title: Analysis title

        Returns:
            Dictionary containing analysis results and visualization paths
        """
        results = {
            "title": title,
            "analysis_summary": {},
            "visualizations": {},
            "data_files": {},
        }

        # Statistical analysis of solution
        results["analysis_summary"] = self._analyze_solution_statistics(
            x_grid, time_grid, density_history, value_history
        )

        # Create visualizations if available
        if self.viz_manager:
            results["visualizations"] = self._create_solution_visualizations(
                x_grid, time_grid, density_history, value_history, title
            )

        # Analyze convergence if provided
        if convergence_data and self.ts_analyzer:
            conv_analysis = self._analyze_convergence(convergence_data)
            results["convergence_analysis"] = conv_analysis

            # Convergence visualizations
            if self.viz_manager:
                conv_viz = self._create_convergence_visualizations(convergence_data, title)
                results["visualizations"].update(conv_viz)

        # Export data if available
        if self.data_exporter:
            data_files = self._export_solution_data(x_grid, time_grid, density_history, value_history, title)
            results["data_files"] = data_files

        logger.info(f"MFG solution analysis completed: {title}")
        return results

    def analyze_parameter_sweep(
        self,
        sweep_results: List[Dict[str, Any]],
        parameter_name: str = "lambda",
        title: str = "Parameter Sweep Analysis",
    ) -> Dict[str, Any]:
        """
        Comprehensive parameter sweep analysis.

        Args:
            sweep_results: List of parameter sweep results
            parameter_name: Name of swept parameter
            title: Analysis title

        Returns:
            Analysis results with visualizations and statistics
        """
        results = {
            "title": title,
            "parameter_name": parameter_name,
            "n_sweeps": len(sweep_results),
            "analysis_summary": {},
            "visualizations": {},
            "data_files": {},
        }

        # Data analysis with Polars
        if self.sweep_analyzer:
            # Create sweep DataFrame
            sweep_df = self.sweep_analyzer.create_sweep_dataframe(sweep_results)

            # Parameter effects analysis
            parameter_cols = [f"param_{parameter_name}"]
            metric_cols = [col for col in sweep_df.columns if col.startswith("metric_")]

            if metric_cols:
                effects = self.sweep_analyzer.analyze_parameter_effects(
                    sweep_df,
                    parameter_cols,
                    metric_cols[:3],  # Limit to first 3 metrics
                )
                results["parameter_effects"] = effects.to_dicts()

            # Find optimal parameters
            if "metric_crater_depth" in sweep_df.columns:
                optimal = self.sweep_analyzer.find_optimal_parameters(sweep_df, "metric_crater_depth", minimize=True)
                results["optimal_parameters"] = optimal

            # Correlation analysis
            correlation_matrix = self.sweep_analyzer.compute_correlation_matrix(sweep_df)
            results["correlations"] = correlation_matrix.to_dicts()

            # Export sweep data
            if self.data_exporter:
                sweep_file = self.output_dir / f"{title.lower().replace(' ', '_')}_sweep_data.parquet"
                self.data_exporter.export_to_parquet(sweep_df, sweep_file)
                results["data_files"]["sweep_data"] = str(sweep_file)

        # Create visualizations
        if self.viz_manager:
            # Interactive dashboard
            dashboard = self.viz_manager.create_parameter_sweep_dashboard(sweep_results, parameter_name, "auto")

            dashboard_file = self.output_dir / f"{title.lower().replace(' ', '_')}_dashboard.html"
            self.viz_manager.save_plot(dashboard, dashboard_file)
            results["visualizations"]["dashboard"] = str(dashboard_file)

        logger.info(f"Parameter sweep analysis completed: {len(sweep_results)} sweeps")
        return results

    def create_research_report(self, analyses: List[Dict[str, Any]], report_title: str = "MFG Research Report") -> Path:
        """
        Generate comprehensive research report combining multiple analyses.

        Args:
            analyses: List of analysis results from analyze_* methods
            report_title: Title for the research report

        Returns:
            Path to generated HTML report
        """
        report_file = self.output_dir / f"{report_title.lower().replace(' ', '_')}.html"

        # Generate comprehensive HTML report
        html_content = self._generate_html_report(analyses, report_title)

        with open(report_file, "w") as f:
            f.write(html_content)

        logger.info(f"Research report generated: {report_file}")
        return report_file

    def _analyze_solution_statistics(
        self,
        x_grid: np.ndarray,
        time_grid: np.ndarray,
        density_history: np.ndarray,
        value_history: np.ndarray,
    ) -> Dict[str, Any]:
        """Analyze statistical properties of MFG solution."""
        stats = {}

        # Density statistics
        final_density = density_history[-1, :]
        stats["density"] = {
            "final_mean": float(np.mean(final_density)),
            "final_std": float(np.std(final_density)),
            "final_max": float(np.max(final_density)),
            "final_min": float(np.min(final_density)),
            "mass_conservation": float(trapezoid(final_density, x=x_grid)),
            "peak_location": float(x_grid[np.argmax(final_density)]),
        }

        # Value function statistics
        final_value = value_history[-1, :]
        stats["value_function"] = {
            "final_mean": float(np.mean(final_value)),
            "final_std": float(np.std(final_value)),
            "final_max": float(np.max(final_value)),
            "final_min": float(np.min(final_value)),
            "gradient_norm": float(np.mean(np.abs(np.gradient(final_value)))),
        }

        # Temporal evolution
        stats["evolution"] = {
            "density_variance_trend": [
                float(np.var(density_history[i, :])) for i in range(0, len(time_grid), max(1, len(time_grid) // 10))
            ],
            "value_range_trend": [
                float(np.max(value_history[i, :]) - np.min(value_history[i, :]))
                for i in range(0, len(time_grid), max(1, len(time_grid) // 10))
            ],
        }

        return stats

    def _create_solution_visualizations(
        self,
        x_grid: np.ndarray,
        time_grid: np.ndarray,
        density_history: np.ndarray,
        value_history: np.ndarray,
        title: str,
    ) -> Dict[str, str]:
        """Create comprehensive solution visualizations."""
        viz_files = {}

        # 2D density evolution
        density_2d = self.viz_manager.create_2d_density_plot(
            x_grid,
            time_grid,
            density_history,
            "auto",
            f"{title}: Density Evolution m(t,x)",
        )
        density_2d_file = self.output_dir / f"{title.lower().replace(' ', '_')}_density_2d.html"
        self.viz_manager.save_plot(density_2d, density_2d_file)
        viz_files["density_2d"] = str(density_2d_file)

        # 3D density surface (if Plotly available)
        if PLOTLY_AVAILABLE:
            density_3d = self.viz_manager.create_3d_surface_plot(
                x_grid,
                time_grid,
                density_history,
                "density",
                f"{title}: 3D Density Surface",
            )
            density_3d_file = self.output_dir / f"{title.lower().replace(' ', '_')}_density_3d.html"
            self.viz_manager.save_plot(density_3d, density_3d_file)
            viz_files["density_3d"] = str(density_3d_file)

            # 3D value function surface
            value_3d = self.viz_manager.create_3d_surface_plot(
                x_grid, time_grid, value_history, "value", f"{title}: 3D Value Function"
            )
            value_3d_file = self.output_dir / f"{title.lower().replace(' ', '_')}_value_3d.html"
            self.viz_manager.save_plot(value_3d, value_3d_file)
            viz_files["value_3d"] = str(value_3d_file)

        return viz_files

    def _analyze_convergence(self, convergence_data: List[Dict]) -> Dict[str, Any]:
        """Analyze convergence properties."""
        if not self.ts_analyzer:
            return {}

        # Create convergence DataFrame
        conv_df = self.ts_analyzer.create_convergence_dataframe(convergence_data)

        # Analyze convergence rate
        conv_stats = self.ts_analyzer.analyze_convergence_rate(conv_df)

        # Detect plateau
        plateau_iter = self.ts_analyzer.detect_convergence_plateau(conv_df)

        return {
            "convergence_statistics": conv_stats,
            "plateau_iteration": plateau_iter,
            "total_iterations": len(convergence_data),
            "final_error": (convergence_data[-1].get("error", 0) if convergence_data else 0),
        }

    def _create_convergence_visualizations(self, convergence_data: List[Dict], title: str) -> Dict[str, str]:
        """Create convergence visualizations."""
        viz_files = {}

        if PLOTLY_AVAILABLE and self.viz_manager.plotly_viz:
            # Convergence animation
            conv_animation = self.viz_manager.plotly_viz.create_convergence_animation(
                convergence_data, f"{title}: Convergence Animation"
            )

            conv_file = self.output_dir / f"{title.lower().replace(' ', '_')}_convergence.html"
            self.viz_manager.save_plot(conv_animation, conv_file)
            viz_files["convergence_animation"] = str(conv_file)

        return viz_files

    def _export_solution_data(
        self,
        x_grid: np.ndarray,
        time_grid: np.ndarray,
        density_history: np.ndarray,
        value_history: np.ndarray,
        title: str,
    ) -> Dict[str, str]:
        """Export solution data in multiple formats."""
        data_files = {}

        if not self.data_exporter:
            return data_files

        # Create comprehensive data structure with consistent dimensions
        # Use spatial grid as the primary dimension
        solution_data = []
        final_density = density_history[-1, :]
        final_value = value_history[-1, :]

        for i, x in enumerate(x_grid):
            solution_data.append(
                {
                    "x_position": float(x),
                    "final_density": float(final_density[i]),
                    "final_value": float(final_value[i]),
                    "density_gradient": (float(np.gradient(final_density)[i]) if len(final_density) > 1 else 0.0),
                }
            )

        # Add metadata as separate records
        metadata = {
            "x_position": -1.0,  # Special marker for metadata
            "final_density": float(np.mean(final_density)),
            "final_value": float(np.mean(final_value)),
            "density_gradient": float(np.mean(np.abs(np.gradient(final_density)))),
        }
        solution_data.append(metadata)

        # Create DataFrame for export
        from ..utils.polars_integration import create_mfg_dataframe

        solution_df = create_mfg_dataframe(solution_data)

        # Export in multiple formats
        base_name = title.lower().replace(" ", "_")

        # Parquet (most efficient)
        parquet_file = self.output_dir / f"{base_name}_solution.parquet"
        self.data_exporter.export_to_parquet(solution_df, parquet_file)
        data_files["parquet"] = str(parquet_file)

        # CSV (human readable)
        csv_file = self.output_dir / f"{base_name}_solution.csv"
        self.data_exporter.export_to_csv(solution_df, csv_file)
        data_files["csv"] = str(csv_file)

        return data_files

    def _generate_html_report(self, analyses: List[Dict[str, Any]], report_title: str) -> str:
        """Generate comprehensive HTML research report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report_title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e90; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        .summary {{ background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .visualization {{ margin: 20px 0; }}
        .statistics {{ background: #f8f9fa; padding: 15px; border-radius: 5px; }}
        .file-link {{ color: #3498db; text-decoration: none; }}
        .file-link:hover {{ text-decoration: underline; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #bdc3c7; padding: 8px; text-align: left; }}
        th {{ background-color: #34495e; color: white; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #e8f6ff; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>{report_title}</h1>
    <div class="summary">
        <h2>Executive Summary</h2>
        <p>This report presents comprehensive analysis of {len(analyses)} MFG studies.</p>
        <p>Generated using the MFG_PDE Advanced Analytics Engine with Polars data processing and Plotly/Bokeh visualizations.</p>
    </div>
"""

        # Add each analysis
        for i, analysis in enumerate(analyses, 1):
            html += f"""
    <h2>Analysis {i}: {analysis.get('title', 'Untitled')}</h2>
    
    <h3>Key Statistics</h3>
    <div class="statistics">
"""

            # Add analysis-specific content
            if "analysis_summary" in analysis:
                summary = analysis["analysis_summary"]
                if "density" in summary:
                    density_stats = summary["density"]
                    html += f"""
        <div class="metric">
            <strong>Final Density Mean:</strong> {density_stats.get('final_mean', 'N/A'):.4f}
        </div>
        <div class="metric">
            <strong>Peak Location:</strong> {density_stats.get('peak_location', 'N/A'):.3f}
        </div>
        <div class="metric">
            <strong>Mass Conservation:</strong> {density_stats.get('mass_conservation', 'N/A'):.4f}
        </div>
"""

            if "n_sweeps" in analysis:
                html += f"""
        <div class="metric">
            <strong>Parameter Sweeps:</strong> {analysis['n_sweeps']}
        </div>
        <div class="metric">
            <strong>Parameter:</strong> {analysis.get('parameter_name', 'N/A')}
        </div>
"""

            html += "</div>"

            # Add visualizations
            if "visualizations" in analysis and analysis["visualizations"]:
                html += "<h3>Visualizations</h3><div class='visualization'>"
                for viz_name, viz_path in analysis["visualizations"].items():
                    viz_filename = Path(viz_path).name
                    html += (
                        f'<p><a href="{viz_filename}" class="file-link"> {viz_name.replace("_", " ").title()}</a></p>'
                    )
                html += "</div>"

            # Add data files
            if "data_files" in analysis and analysis["data_files"]:
                html += "<h3>Data Files</h3><div class='visualization'>"
                for file_name, file_path in analysis["data_files"].items():
                    filename = Path(file_path).name
                    html += (
                        f'<p><a href="{filename}" class="file-link">💾 {file_name.replace("_", " ").title()}</a></p>'
                    )
                html += "</div>"

        html += """
    <hr>
    <footer>
        <p><em>Generated by MFG_PDE Advanced Analytics Engine</em></p>
        <p>Features: High-performance Polars data processing, Interactive Plotly/Bokeh visualizations</p>
    </footer>
</body>
</html>"""

        return html


# Factory function
def create_analytics_engine(
    prefer_plotly: bool = True, output_dir: Optional[Union[str, Path]] = None
) -> MFGAnalyticsEngine:
    """Create MFG Analytics Engine instance."""
    return MFGAnalyticsEngine(prefer_plotly, output_dir)


# Convenience functions
def analyze_mfg_solution_quick(
    x_grid: np.ndarray,
    time_grid: np.ndarray,
    density_history: np.ndarray,
    value_history: np.ndarray,
    title: str = "MFG Analysis",
) -> Dict[str, Any]:
    """Quick MFG solution analysis with default settings."""
    engine = create_analytics_engine()
    return engine.analyze_mfg_solution(x_grid, time_grid, density_history, value_history, title=title)


def analyze_parameter_sweep_quick(
    sweep_results: List[Dict[str, Any]],
    parameter_name: str = "lambda",
    title: str = "Parameter Sweep",
) -> Dict[str, Any]:
    """Quick parameter sweep analysis with default settings."""
    engine = create_analytics_engine()
    return engine.analyze_parameter_sweep(sweep_results, parameter_name, title)
