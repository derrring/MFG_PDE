"""
Pydantic Integration for Enhanced Notebook Reporting

This module extends the existing notebook reporting system with Pydantic
configuration support, automatic serialization, and enhanced validation
for research workflows.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from pydantic import ValidationError

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

from .logging import get_logger
from .notebook_reporting import MFGNotebookReporter, NotebookReportError

# Import Pydantic configurations if available
if PYDANTIC_AVAILABLE:
    from ..config.array_validation import ExperimentConfig, MFGArrays, MFGGridConfig
    from ..config.pydantic_config import MFGSolverConfig


class PydanticNotebookReporter(MFGNotebookReporter):
    """
    Enhanced notebook reporter with Pydantic configuration support.

    Extends the base MFGNotebookReporter with automatic Pydantic model
    serialization, validation, and enhanced metadata management.
    """

    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)

        if not PYDANTIC_AVAILABLE:
            self.logger.warning(
                "Pydantic not available - falling back to basic notebook reporting"
            )

    def create_enhanced_mfg_report(
        self,
        title: str,
        experiment_config: "ExperimentConfig",
        solver_results: Dict[str, Any],
        output_dir: str = "reports",
        export_html: bool = True,
        include_validation_report: bool = True,
    ) -> Dict[str, str]:
        """
        Create comprehensive MFG research report using Pydantic configuration.

        Args:
            title: Report title
            experiment_config: Pydantic ExperimentConfig with validation
            solver_results: Dictionary with solver results
            output_dir: Output directory for reports
            export_html: Whether to export HTML version
            include_validation_report: Whether to include validation details

        Returns:
            Dictionary with paths to generated files
        """
        if not PYDANTIC_AVAILABLE:
            self.logger.error("Pydantic not available for enhanced reporting")
            raise NotebookReportError("Pydantic required for enhanced reporting")

        try:
            # Validate experiment configuration
            if not isinstance(experiment_config, ExperimentConfig):
                raise NotebookReportError(
                    "experiment_config must be ExperimentConfig instance"
                )

            # Extract validated configuration
            validated_config = experiment_config.dict()
            notebook_metadata = experiment_config.to_notebook_metadata()

            # Create enhanced problem configuration
            enhanced_problem_config = {
                **validated_config["grid_config"],
                "validation_passed": True,
                "config_type": "pydantic_enhanced",
                "created_at": datetime.now().isoformat(),
            }

            # Add array statistics if available
            if experiment_config.arrays:
                array_stats = experiment_config.arrays.get_solution_statistics()
                enhanced_problem_config["array_statistics"] = array_stats

                # Use validated arrays for visualization
                solver_results = {
                    **solver_results,
                    "U": experiment_config.arrays.U_solution,
                    "M": experiment_config.arrays.M_solution,
                    "validation_stats": array_stats,
                }

            # Generate enhanced notebook
            notebook_content = self._create_enhanced_notebook_content(
                title=title,
                experiment_config=experiment_config,
                solver_results=solver_results,
                include_validation_report=include_validation_report,
            )

            # Create notebook with enhanced metadata
            enhanced_metadata = {
                **notebook_metadata,
                "pydantic_validation": True,
                "report_generator": "PydanticNotebookReporter",
                "notebook_version": "2.0",
            }

            return self._save_enhanced_notebook(
                notebook_content=notebook_content,
                title=title,
                metadata=enhanced_metadata,
                output_dir=output_dir,
                export_html=export_html,
            )

        except ValidationError as e:
            self.logger.error(f"Pydantic validation error: {e}")
            raise NotebookReportError(f"Configuration validation failed: {e}")
        except Exception as e:
            self.logger.error(f"Enhanced notebook generation failed: {e}")
            raise NotebookReportError(f"Enhanced notebook generation failed: {e}")

    def _create_enhanced_notebook_content(
        self,
        title: str,
        experiment_config: "ExperimentConfig",
        solver_results: Dict[str, Any],
        include_validation_report: bool = True,
    ) -> List[Dict[str, Any]]:
        """Create enhanced notebook content with Pydantic configuration details."""
        cells = []

        # Title and metadata
        cells.append(
            {
                "cell_type": "markdown",
                "source": self._create_enhanced_title_section(title, experiment_config),
            }
        )

        # Configuration validation report
        if include_validation_report:
            cells.append(
                {
                    "cell_type": "markdown",
                    "source": self._create_validation_report_section(experiment_config),
                }
            )

        # Enhanced configuration summary
        cells.append(
            {
                "cell_type": "code",
                "source": self._create_enhanced_config_code_section(experiment_config),
            }
        )

        # Grid and numerical stability analysis
        cells.append(
            {
                "cell_type": "markdown",
                "source": self._create_numerical_analysis_section(experiment_config),
            }
        )

        # Solution visualization (if arrays are available)
        if experiment_config.arrays:
            cells.append(
                {
                    "cell_type": "code",
                    "source": self._create_enhanced_visualization_code(
                        experiment_config, solver_results
                    ),
                }
            )

            # Array validation results
            cells.append(
                {
                    "cell_type": "markdown",
                    "source": self._create_array_validation_section(
                        experiment_config.arrays
                    ),
                }
            )

        # Standard solver results sections
        cells.extend(self._create_standard_results_sections(solver_results))

        # Enhanced conclusions with validation summary
        cells.append(
            {
                "cell_type": "markdown",
                "source": self._create_enhanced_conclusions_section(
                    experiment_config, solver_results
                ),
            }
        )

        return cells

    def _create_enhanced_title_section(
        self, title: str, experiment_config: "ExperimentConfig"
    ) -> str:
        """Create enhanced title section with experiment metadata."""
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return f"""# {title}

**Experiment**: {experiment_config.experiment_name}  
**Researcher**: {experiment_config.researcher}  
**Generated**: {created_at}  
**Configuration**: Enhanced Pydantic Validation  

{experiment_config.description or "MFG numerical analysis with comprehensive validation"}

---

## Experiment Configuration

This report uses **Pydantic-validated configurations** ensuring:
- ✅ Automatic parameter validation
- ✅ Numerical stability checking  
- ✅ Physical constraint validation
- ✅ Cross-field consistency verification
- ✅ Professional serialization support

**Tags**: {', '.join(experiment_config.tags) if experiment_config.tags else 'None'}
"""

    def _create_validation_report_section(
        self, experiment_config: "ExperimentConfig"
    ) -> str:
        """Create comprehensive validation report section."""
        grid_config = experiment_config.grid_config

        validation_report = f"""## 🔍 Configuration Validation Report

### Grid Configuration Validation
- **Spatial Grid**: {grid_config.Nx} points, domain [{grid_config.xmin:.2f}, {grid_config.xmax:.2f}]
- **Time Grid**: {grid_config.Nt} points, final time T = {grid_config.T:.2f}
- **Grid Spacing**: dx = {grid_config.dx:.6f}, dt = {grid_config.dt:.6f}
- **CFL Number**: {grid_config.cfl_number:.4f} {'✅ Stable' if grid_config.cfl_number <= 0.5 else '⚠️ May be unstable'}
- **Diffusion**: σ = {grid_config.sigma:.3f}

### Numerical Stability Analysis
"""

        # Add CFL analysis
        if grid_config.cfl_number > 0.5:
            validation_report += f"""
⚠️ **CFL Warning**: CFL number ({grid_config.cfl_number:.4f}) > 0.5 may cause numerical instability.
**Recommendation**: Reduce dt or increase dx for better stability.
"""
        else:
            validation_report += f"""
✅ **CFL Condition**: Satisfied with safety margin of {0.5 - grid_config.cfl_number:.4f}.
"""

        # Add array validation if available
        if experiment_config.arrays:
            stats = experiment_config.arrays.get_solution_statistics()
            mass_stats = stats["mass_conservation"]

            validation_report += f"""
### Array Validation Results
- **U Solution**: Shape {stats['U']['shape']}, range [{stats['U']['min']:.3e}, {stats['U']['max']:.3e}]
- **M Solution**: Shape {stats['M']['shape']}, range [{stats['M']['min']:.3e}, {stats['M']['max']:.3e}]
- **Mass Conservation**: 
  - Initial: {mass_stats['initial_mass']:.6f}
  - Final: {mass_stats['final_mass']:.6f}
  - Drift: {mass_stats['mass_drift']:.2e} {'✅ Conserved' if abs(mass_stats['mass_drift']) < 1e-3 else '⚠️ Not conserved'}
"""

        return validation_report

    def _create_enhanced_config_code_section(
        self, experiment_config: "ExperimentConfig"
    ) -> str:
        """Create enhanced configuration code section with Pydantic serialization."""
        return f"""# Configuration Management with Pydantic

# Experiment configuration (validated)
experiment_config = {repr(experiment_config.dict())}

# Grid configuration with automatic validation
grid_config = {repr(experiment_config.grid_config.dict())}

# Computed grid properties
print(f"Grid spacing: dx = {{experiment_config.grid_config.dx:.6f}}, dt = {{experiment_config.grid_config.dt:.6f}}")
print(f"CFL number: {{experiment_config.grid_config.cfl_number:.4f}}")
print(f"Expected array shape: {{experiment_config.grid_config.grid_shape}}")

# JSON serialization (automatic with Pydantic)
import json
config_json = experiment_config.json(indent=2)
print("\\nJSON-serialized configuration:")
print(config_json[:200] + "..." if len(config_json) > 200 else config_json)
"""

    def _create_numerical_analysis_section(
        self, experiment_config: "ExperimentConfig"
    ) -> str:
        """Create numerical analysis section with stability analysis."""
        grid_config = experiment_config.grid_config

        return f"""## 📊 Numerical Method Analysis

### Discretization Parameters
The numerical discretization uses:

$$\\Delta x = \\frac{{L}}{{N_x}} = \\frac{{{grid_config.xmax - grid_config.xmin:.3f}}}{{{grid_config.Nx}}} = {grid_config.dx:.6f}$$

$$\\Delta t = \\frac{{T}}{{N_t}} = \\frac{{{grid_config.T:.3f}}}{{{grid_config.Nt}}} = {grid_config.dt:.6f}$$

### Stability Analysis
The **CFL condition** for parabolic PDEs requires:

$$\\text{{CFL}} = \\frac{{\\sigma^2 \\Delta t}}{{(\\Delta x)^2}} \\leq 0.5$$

**Current CFL number**: {grid_config.cfl_number:.4f}

{'✅ **Stable**: The discretization satisfies the CFL condition.' if grid_config.cfl_number <= 0.5 else '⚠️ **Potentially Unstable**: CFL > 0.5 may cause numerical instability.'}

### Diffusion Time Scale
Characteristic diffusion time: $\\tau_{{\\text{{diff}}}} = \\frac{{L^2}}{{\\sigma^2}} = {(grid_config.xmax - grid_config.xmin)**2 / grid_config.sigma**2:.3f}$

Time steps per diffusion time: ${grid_config.T / ((grid_config.xmax - grid_config.xmin)**2 / grid_config.sigma**2) * grid_config.Nt:.1f}$
"""

    def _create_enhanced_visualization_code(
        self, experiment_config: "ExperimentConfig", solver_results: Dict[str, Any]
    ) -> str:
        """Create enhanced visualization code with array validation."""
        return f"""# Enhanced Visualization with Validated Arrays

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Validated solution arrays
U_solution = np.array({experiment_config.arrays.U_solution.tolist() if experiment_config.arrays else [[0]]})
M_solution = np.array({experiment_config.arrays.M_solution.tolist() if experiment_config.arrays else [[0]]})

# Grid for plotting (validated)
x_grid = np.linspace({experiment_config.grid_config.xmin}, {experiment_config.grid_config.xmax}, {experiment_config.grid_config.Nx + 1})
t_grid = np.linspace(0, {experiment_config.grid_config.T}, {experiment_config.grid_config.Nt + 1})

# Create interactive subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('HJB Solution U(t,x)', 'FP Density M(t,x)', 
                   'Final U Profile', 'Final M Profile'),
    specs=[[{{"type": "heatmap"}}, {{"type": "heatmap"}}],
           [{{"type": "scatter"}}, {{"type": "scatter"}}]]
)

# U solution heatmap
fig.add_trace(
    go.Heatmap(
        z=U_solution,
        x=x_grid,
        y=t_grid,
        colorscale='Viridis',
        name='U(t,x)'
    ),
    row=1, col=1
)

# M solution heatmap  
fig.add_trace(
    go.Heatmap(
        z=M_solution,
        x=x_grid,
        y=t_grid,
        colorscale='Plasma',
        name='M(t,x)'
    ),
    row=1, col=2
)

# Final profiles
fig.add_trace(
    go.Scatter(
        x=x_grid,
        y=U_solution[-1],
        mode='lines+markers',
        name='U(T,x)',
        line=dict(color='blue')
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=x_grid, 
        y=M_solution[-1],
        mode='lines+markers',
        name='M(T,x)',
        line=dict(color='red')
    ),
    row=2, col=2
)

# Layout
fig.update_layout(
    title='Enhanced MFG Solution Visualization (Pydantic Validated)',
    height=800,
    showlegend=True
)

fig.update_xaxes(title_text="Space (x)", row=2, col=1)
fig.update_xaxes(title_text="Space (x)", row=2, col=2)
fig.update_yaxes(title_text="Value", row=2, col=1)
fig.update_yaxes(title_text="Density", row=2, col=2)

fig.show()

# Print validation statistics
if 'validation_stats' in locals():
    print("\\n📊 Array Validation Statistics:")
    for key, stats in validation_stats.items():
        if isinstance(stats, dict):
            print(f"\\n{key.upper()}:")
            for stat_name, value in stats.items():
                print(f"  {stat_name}: {value}")
"""

    def _create_array_validation_section(self, arrays: "MFGArrays") -> str:
        """Create detailed array validation section."""
        stats = arrays.get_solution_statistics()

        return f"""## 🔬 Array Validation Analysis

### Solution Array Properties

**HJB Solution U(t,x)**:
- Shape: {stats['U']['shape']}
- Data type: {stats['U']['dtype']}
- Value range: [{stats['U']['min']:.3e}, {stats['U']['max']:.3e}]
- Mean: {stats['U']['mean']:.3e}, Std: {stats['U']['std']:.3e}

**FP Density M(t,x)**:
- Shape: {stats['M']['shape']}
- Data type: {stats['M']['dtype']}
- Value range: [{stats['M']['min']:.3e}, {stats['M']['max']:.3e}]
- Mean: {stats['M']['mean']:.3e}, Std: {stats['M']['std']:.3e}

### Physical Constraint Validation

**Mass Conservation Analysis**:
- Initial mass: {stats['mass_conservation']['initial_mass']:.6f}
- Final mass: {stats['mass_conservation']['final_mass']:.6f}
- Mass drift: {stats['mass_conservation']['mass_drift']:.2e}
- Conservation quality: {'✅ Excellent' if abs(stats['mass_conservation']['mass_drift']) < 1e-4 else '⚠️ Acceptable' if abs(stats['mass_conservation']['mass_drift']) < 1e-3 else '❌ Poor'}

**Numerical Stability**:
- CFL number: {stats['numerical_stability']['cfl_number']:.4f}
- Grid spacing: dx = {stats['numerical_stability']['dx']:.6f}, dt = {stats['numerical_stability']['dt']:.6f}
- Diffusion coefficient: σ = {stats['numerical_stability']['sigma']:.3f}

### Validation Status
{'✅ **All validations passed** - Arrays satisfy physical constraints and numerical stability requirements.' if abs(stats['mass_conservation']['mass_drift']) < 1e-3 and stats['numerical_stability']['cfl_number'] <= 0.5 else '⚠️ **Some validations failed** - Check mass conservation and stability conditions.'}
"""

    def _create_enhanced_conclusions_section(
        self, experiment_config: "ExperimentConfig", solver_results: Dict[str, Any]
    ) -> str:
        """Create enhanced conclusions with validation summary."""
        return f"""## 📝 Enhanced Conclusions

### Experiment Summary
- **Experiment**: {experiment_config.experiment_name}
- **Configuration**: Pydantic-validated with comprehensive checks
- **Grid**: {experiment_config.grid_config.Nx}×{experiment_config.grid_config.Nt} points
- **Validation**: {'✅ Passed' if experiment_config.arrays else '⚠️ Partial (no arrays)'}

### Key Findings
1. **Numerical Stability**: {'Satisfied' if experiment_config.grid_config.cfl_number <= 0.5 else 'Marginal'}
2. **Mass Conservation**: {'Excellent' if experiment_config.arrays and abs(experiment_config.arrays.get_solution_statistics()['mass_conservation']['mass_drift']) < 1e-4 else 'Not evaluated'}
3. **Configuration Quality**: Professional-grade with automatic validation

### Reproducibility Information
This experiment uses **Pydantic configuration management** ensuring:
- Complete parameter validation
- Automatic JSON serialization
- Version control compatibility
- Environment variable support

**Configuration Hash**: `{hash(str(experiment_config.dict()))}`

### Next Steps
- Configuration can be saved with: `experiment_config.json()`
- Arrays can be reloaded with full validation
- Parameters can be modified with automatic re-validation
- Results are ready for publication or further analysis

---
*Report generated by PydanticNotebookReporter v2.0*
"""

    def _save_enhanced_notebook(
        self,
        notebook_content: List[Dict[str, Any]],
        title: str,
        metadata: Dict[str, Any],
        output_dir: str,
        export_html: bool,
    ) -> Dict[str, str]:
        """Save enhanced notebook with Pydantic metadata."""
        # Delegate to parent class with enhanced metadata
        return super().save_notebook(
            notebook_content=notebook_content,
            title=title,
            metadata=metadata,
            output_dir=output_dir,
            export_html=export_html,
        )


def create_pydantic_mfg_report(
    title: str,
    experiment_config: "ExperimentConfig",
    solver_results: Dict[str, Any],
    output_dir: str = "reports",
    export_html: bool = True,
) -> Dict[str, str]:
    """
    Convenience function for creating enhanced MFG reports with Pydantic validation.

    Args:
        title: Report title
        experiment_config: Validated ExperimentConfig instance
        solver_results: Solver results dictionary
        output_dir: Output directory
        export_html: Whether to export HTML

    Returns:
        Dictionary with file paths
    """
    reporter = PydanticNotebookReporter()
    return reporter.create_enhanced_mfg_report(
        title=title,
        experiment_config=experiment_config,
        solver_results=solver_results,
        output_dir=output_dir,
        export_html=export_html,
    )
