# Notebook Export Enhancement Plan
**Created**: 2025-10-11
**Status**: 🔄 Planning
**Related Issue**: TBD
**Priority**: Medium
**Size**: Small-Medium (~3.5 hours)

## Context

Currently, `SolverResult.export_summary()` supports only `markdown` and `latex` output formats. However, MFG_PDE has comprehensive Jupyter notebook support through `mfg_pde.utils.notebooks.reporting.MFGNotebookReporter`.

**User Request**: Add notebook export support to `export_summary()` method to enable interactive exploration of solver results.

## Existing Infrastructure

### Available Assets
- ✅ **`MFGNotebookReporter`** (`mfg_pde/utils/notebooks/reporting.py`, 973 lines)
  - Complete notebook generation system
  - Plotly integration for interactive visualizations
  - LaTeX math support
  - HTML export capability
  - Professional research report templates

- ✅ **`nbformat` dependency** (already in requirements)
  - Core notebook creation capability
  - Cell manipulation (markdown, code)
  - Notebook validation

- ✅ **Current `export_summary()` methods**
  - `_export_markdown()` - Clean markdown summaries
  - `_export_latex()` - Publication-ready LaTeX tables
  - File writing with pathlib

### Current Limitations
- ❌ No notebook format support in `export_summary()`
- ❌ No integration between `SolverResult` and `MFGNotebookReporter`
- ❌ Static exports only (no interactive exploration)

## Proposed Solution

### Two-Phase Approach ⚡ **BOTH PHASES APPROVED**

#### **Phase 1: Simple Template Export** ✅ IMMEDIATE
**Scope**: Add `"notebook"` format to `export_summary()` with code templates

**Purpose**: Quick, lightweight summaries for reference and documentation

**Implementation**:
1. Add `"notebook"` to supported formats list
2. Implement `_export_notebook()` private method
3. Create notebook with structured cells:
   - **Cell 1**: Markdown title and summary (from `_export_markdown()`)
   - **Cell 2**: Code to recreate `SolverResult` from exported data
   - **Cell 3**: Code demonstrating `analyze_convergence()`
   - **Cell 4**: Code demonstrating `plot_convergence()`
   - **Cell 5**: Code template for `compare_to()` usage
   - **Cell 6**: Markdown with export instructions

**Benefits**:
- ✅ Simple, focused enhancement (~3.5 hours)
- ✅ Reuses existing infrastructure
- ✅ Maintains API consistency
- ✅ Enables immediate interactive value
- ✅ No breaking changes

**API Example**:
```python
from mfg_pde import solve_mfg

result = solve_mfg(problem)

# New capability - exports interactive notebook
result.export_summary(
    output_format="notebook",
    filename="analysis.ipynb"
)

# Existing capabilities unchanged
result.export_summary(output_format="markdown", filename="summary.md")
result.export_summary(output_format="latex", filename="summary.tex")
```

**Generated Notebook Structure**:
```python
# Cell 1 (markdown):
"""
# Solver Results: HJB-FP Solver

## Summary
- Status: ✅ Converged
- Iterations: 50
- Final Error (U): 7.45e-03
...
"""

# Cell 2 (code):
import numpy as np
from mfg_pde.utils.solver_result import SolverResult

# Recreate SolverResult for interactive exploration
result = SolverResult(
    U=np.load("exported_U.npy"),  # Or embedded data
    M=np.load("exported_M.npy"),
    ...
)

# Cell 3 (code):
# Convergence Analysis
analysis = result.analyze_convergence()
print(f"Convergence Rate: {analysis.convergence_rate:.4f}")
print(f"Stagnation Detected: {analysis.stagnation_detected}")
print(f"Oscillation Detected: {analysis.oscillation_detected}")

# Cell 4 (code):
# Convergence Plot
result.plot_convergence(
    save_path="convergence.png",
    show=True,
    log_scale=True
)

# Cell 5 (code):
# Solver Comparison Template
# result2 = solve_mfg(problem, different_config)
# comparison = result.compare_to(result2)
# print(f"Solution Difference: {comparison.solution_diff_l2:.6e}")

# Cell 6 (markdown):
"""
## Export and Sharing
- HTML: `jupyter nbconvert --to html analysis.ipynb`
- PDF: `jupyter nbconvert --to pdf analysis.ipynb`
- Execute: `jupyter nbconvert --execute analysis.ipynb`
"""
```

#### **Phase 2: Convenience Wrapper for Rich Reports** ✅ APPROVED
**Scope**: Add `create_research_report()` convenience method to `SolverResult` that wraps `MFGNotebookReporter`

**Purpose**: Publication-quality reports with interactive visualizations and comprehensive analysis

**Implementation**:
1. Add `create_research_report()` method to `SolverResult` class
2. Method internally:
   - Constructs data dictionaries from SolverResult attributes
   - Calls `MFGNotebookReporter.create_research_report()`
   - Returns notebook path (and optionally HTML path)
3. Provides simple API while leveraging full reporter power

**Benefits**:
- ✅ **Both options available**: Simple templates OR rich reports
- ✅ **Simple API**: User doesn't build dictionaries manually
- ✅ **Leverages existing code**: No duplication of MFGNotebookReporter
- ✅ **Comprehensive reports**: Interactive Plotly, LaTeX math, full analysis
- ✅ **HTML export**: Automatic shareable reports

**API Example**:
```python
result = solve_mfg(problem)

# Option 1: Quick summary (Phase 1)
result.export_summary(format="notebook", filename="summary.ipynb")

# Option 2: Rich research report (Phase 2)
paths = result.create_research_report(
    title="MFG Analysis Report",
    problem_config=config,  # User provides config
    output_dir="reports",
    export_html=True
)
# Returns: {"notebook": "path.ipynb", "html": "path.html"}
```

**Decision**: ✅ **Implement after Phase 1** - Provides complete solution for both use cases

## Implementation Plan

### Phase 1: Simple Template Export

#### 1. Core Implementation
**File**: `mfg_pde/utils/solver_result.py`

**Changes**:
```python
def export_summary(
    self,
    output_format: str = "markdown",
    filename: str | Path | None = None,
) -> str:
    """
    Generate publication-ready summary of solver results.

    Args:
        output_format: Output format ('markdown', 'latex', or 'notebook')
        filename: Optional path to save summary

    Returns:
        Formatted summary string (or notebook path for notebook format)
    """
    if output_format not in ("markdown", "latex", "notebook"):
        raise ValueError(
            f"Unsupported format: {output_format}. "
            f"Use 'markdown', 'latex', or 'notebook'"
        )

    if output_format == "markdown":
        summary = self._export_markdown()
    elif output_format == "latex":
        summary = self._export_latex()
    else:
        # notebook format - different return behavior
        return self._export_notebook(filename)

    # Save markdown/latex if filename provided
    if filename is not None:
        Path(filename).write_text(summary)

    return summary

def _export_notebook(self, filename: str | Path | None = None) -> str:
    """Generate interactive Jupyter notebook summary."""
    try:
        import nbformat as nbf
        from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
    except ImportError:
        raise ImportError(
            "Jupyter notebook export requires nbformat. "
            "Install with: pip install nbformat jupyter"
        )

    # Create notebook
    nb = new_notebook()

    # Cell 1: Title and summary (markdown)
    nb.cells.append(new_markdown_cell(self._export_markdown()))

    # Cell 2: Setup code
    setup_code = """# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from mfg_pde.utils.solver_result import SolverResult

# Note: This notebook demonstrates analysis methods.
# To recreate the full result, you need the original data.
"""
    nb.cells.append(new_code_cell(setup_code))

    # Cell 3: Convergence analysis demo
    analysis_code = f"""# Convergence Analysis
# If you have the SolverResult object:
# analysis = result.analyze_convergence()
# print(f"Convergence Rate: {{analysis.convergence_rate:.4f}}")
# print(f"Stagnation: {{analysis.stagnation_detected}}")
# print(f"Oscillation: {{analysis.oscillation_detected}}")

# For this exported result:
print("Solver: {self.solver_name}")
print(f"Converged: {self.converged}")
print(f"Iterations: {self.iterations}")
print(f"Final Error (U): {self.final_error_U:.6e}")
print(f"Final Error (M): {self.final_error_M:.6e}")
"""
    nb.cells.append(new_code_cell(analysis_code))

    # Cell 4: Plotting demo
    plot_code = """# Convergence Plot
# If you have the SolverResult object:
# result.plot_convergence(save_path='convergence.png', show=True)

# To recreate manually:
# plt.figure(figsize=(10, 6))
# plt.semilogy(range(iterations), error_history_U, 'o-', label='U error')
# plt.semilogy(range(iterations), error_history_M, 's-', label='M error')
# plt.xlabel('Iteration')
# plt.ylabel('Error (L∞ norm)')
# plt.legend()
# plt.grid(True)
# plt.show()
"""
    nb.cells.append(new_code_cell(plot_code))

    # Cell 5: Comparison template
    comparison_code = """# Solver Comparison Template
# Compare this result with another solver:
# result2 = solve_mfg(problem, different_config)
# comparison = result.compare_to(result2)
#
# print(f"Solution Difference (L2): {comparison.solution_diff_l2:.6e}")
# print(f"Solution Difference (L∞): {comparison.solution_diff_linf:.6e}")
# print(f"Faster Solver: {comparison.faster_solver}")
# print(f"More Accurate: {comparison.more_accurate_solver}")
"""
    nb.cells.append(new_code_cell(comparison_code))

    # Cell 6: Export instructions
    export_md = """## Export and Sharing

This notebook can be exported in multiple formats:

- **HTML** (with plots): `jupyter nbconvert --to html --execute notebook.ipynb`
- **PDF** (static): `jupyter nbconvert --to pdf notebook.ipynb`
- **Slides**: `jupyter nbconvert --to slides --post serve notebook.ipynb`
- **Python script**: `jupyter nbconvert --to python notebook.ipynb`

### Interactive Usage

For full interactive analysis, recreate the `SolverResult` object with your data:

```python
result = SolverResult(U=U, M=M, iterations=iterations, ...)
analysis = result.analyze_convergence()
result.plot_convergence()
```
"""
    nb.cells.append(new_markdown_cell(export_md))

    # Save notebook
    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"solver_result_{self.solver_name}_{timestamp}.ipynb"

    filepath = Path(filename)
    with open(filepath, "w") as f:
        nbf.write(nb, f)

    return str(filepath)
```

**Estimated Lines**: ~100 lines

#### 2. Testing (Phase 1)
**File**: `tests/unit/test_utils/test_solver_result_analysis.py`

**New Tests**:
```python
class TestNotebookExport:
    """Tests for notebook export functionality."""

    def test_notebook_export_creates_file(self, converged_result, tmp_path):
        """Test that notebook export creates valid .ipynb file."""
        save_path = tmp_path / "test_export.ipynb"
        result_path = converged_result.export_summary(
            output_format="notebook",
            filename=save_path
        )

        assert Path(result_path).exists()
        assert Path(result_path).suffix == ".ipynb"

    def test_notebook_export_valid_structure(self, converged_result, tmp_path):
        """Test that exported notebook has valid structure."""
        save_path = tmp_path / "test_export.ipynb"
        converged_result.export_summary(
            output_format="notebook",
            filename=save_path
        )

        import nbformat as nbf
        with open(save_path) as f:
            nb = nbf.read(f, as_version=4)

        # Should have at least 6 cells
        assert len(nb.cells) >= 6

        # First cell should be markdown
        assert nb.cells[0].cell_type == "markdown"
        assert "Solver Results" in nb.cells[0].source

        # Subsequent cells should be code
        assert nb.cells[1].cell_type == "code"

    def test_notebook_export_without_nbformat(self, converged_result, tmp_path, monkeypatch):
        """Test graceful handling when nbformat unavailable."""
        # Mock nbformat import to fail
        import sys
        monkeypatch.setitem(sys.modules, "nbformat", None)

        with pytest.raises(ImportError, match="nbformat"):
            converged_result.export_summary(
                output_format="notebook",
                filename=tmp_path / "test.ipynb"
            )

    def test_unsupported_format_error(self, converged_result):
        """Test error for unsupported format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            converged_result.export_summary(output_format="pdf")
```

**Estimated Lines**: ~60 lines

#### 3. Documentation (Phase 1)
**File**: `examples/basic/solver_result_analysis_demo.py`

**Add New Demo Function**:
```python
def demo_export_notebook():
    """Demonstrate notebook export."""
    print("\n" + "=" * 60)
    print("Demo 5: Notebook Export")
    print("=" * 60)

    # Create result
    result = create_sample_result("Demo Solver", converged=True)

    # Export notebook
    nb_path = OUTPUT_DIR / "solver_analysis.ipynb"
    print(f"\nExporting interactive notebook: {nb_path}")
    result.export_summary(output_format="notebook", filename=nb_path)
    print(f"✅ Notebook saved to: {nb_path}")
    print("\nYou can now:")
    print("  1. Open it: jupyter notebook", nb_path.name)
    print("  2. Execute: jupyter nbconvert --execute --to html", nb_path.name)
```

**Update main()**:
```python
def main():
    """Run all demonstrations."""
    # ... existing demos ...
    demo_export_notebook()  # Add new demo
    # ...
```

#### 4. Update README (Phase 1)
**File**: `README.md` (or relevant user guide)

**Add Section**:
```markdown
#### Export as Interactive Notebook

Export solver results as Jupyter notebooks for interactive exploration:

```python
result = solve_mfg(problem)

# Export as interactive notebook
result.export_summary(
    output_format="notebook",
    filename="analysis.ipynb"
)

# Open and explore interactively
# jupyter notebook analysis.ipynb
```

The generated notebook includes:
- Summary of solver results
- Interactive convergence analysis
- Plotting code templates
- Comparison templates
```

## Dependencies

**Required**:
- `nbformat` - Already in package dependencies
- `jupyter` - Optional (for viewing notebooks)

**Optional**:
- `plotly` - For Phase 2 (rich visualizations)
- `kaleido` - For Phase 2 (static image export)

### Phase 2: Rich Report Convenience Wrapper

#### 1. Core Implementation (Phase 2)
**File**: `mfg_pde/utils/solver_result.py`

**Add Method**:
```python
def create_research_report(
    self,
    title: str,
    problem_config: dict[str, Any],
    output_dir: str = "reports",
    analysis_metadata: dict[str, Any] | None = None,
    export_html: bool = True,
) -> dict[str, str]:
    """
    Create comprehensive research report using MFGNotebookReporter.

    This generates a publication-quality Jupyter notebook with:
    - Mathematical framework (LaTeX equations)
    - Interactive Plotly visualizations
    - Convergence analysis
    - Mass conservation analysis
    - HTML export for sharing

    Args:
        title: Report title
        problem_config: MFG problem configuration dictionary
        output_dir: Output directory for reports
        analysis_metadata: Additional metadata for context
        export_html: Whether to export HTML version

    Returns:
        Dictionary with 'notebook' and optionally 'html' paths

    Example:
        >>> result = solver.solve()
        >>> paths = result.create_research_report(
        ...     title="LQ-MFG Analysis",
        ...     problem_config={"sigma": 0.5, "T": 1.0},
        ...     export_html=True
        ... )
        >>> print(f"Notebook: {paths['notebook']}")
        >>> print(f"HTML: {paths['html']}")

    Note:
        Requires plotly for interactive visualizations.
        For simple summaries, use export_summary(format='notebook').
    """
    try:
        from mfg_pde.utils.notebooks.reporting import create_mfg_research_report
    except ImportError:
        raise ImportError(
            "Research reports require notebook support. "
            "Install with: pip install plotly nbformat jupyter"
        )

    # Construct solver_results dictionary from SolverResult attributes
    solver_results = {
        "U": self.U,
        "M": self.M,
        "convergence_info": {
            "converged": self.converged,
            "iterations": self.iterations,
            "error_history": list(self.error_history_U),  # Convert to list
            "final_error": self.max_error,
        },
    }

    # Add execution time if available
    if self.execution_time is not None:
        solver_results["execution_time"] = self.execution_time

    # Add metadata from SolverResult
    if self.metadata:
        solver_results["metadata"] = self.metadata

    # Create report using existing infrastructure
    return create_mfg_research_report(
        title=title,
        solver_results=solver_results,
        problem_config=problem_config,
        output_dir=output_dir,
        export_html=export_html,
    )
```

**Estimated Lines**: ~70 lines

#### 2. Testing (Phase 2)
**File**: `tests/unit/test_utils/test_solver_result_analysis.py`

**New Tests**:
```python
class TestResearchReportCreation:
    """Tests for create_research_report() method."""

    def test_research_report_creates_notebook(self, converged_result, tmp_path):
        """Test that research report creates notebook file."""
        pytest.importorskip("plotly")  # Skip if plotly not available

        paths = converged_result.create_research_report(
            title="Test Report",
            problem_config={"sigma": 0.5, "T": 1.0},
            output_dir=str(tmp_path),
            export_html=False
        )

        assert "notebook" in paths
        assert Path(paths["notebook"]).exists()
        assert Path(paths["notebook"]).suffix == ".ipynb"

    def test_research_report_with_html_export(self, converged_result, tmp_path):
        """Test that HTML export works when enabled."""
        pytest.importorskip("plotly")

        paths = converged_result.create_research_report(
            title="Test Report",
            problem_config={"sigma": 0.5},
            output_dir=str(tmp_path),
            export_html=True
        )

        # HTML might fail gracefully, so check if it exists
        if "html" in paths:
            assert Path(paths["html"]).exists()

    def test_research_report_notebook_structure(self, converged_result, tmp_path):
        """Test that generated notebook has comprehensive structure."""
        pytest.importorskip("plotly")

        paths = converged_result.create_research_report(
            title="Test Report",
            problem_config={"sigma": 0.5},
            output_dir=str(tmp_path),
            export_html=False
        )

        import nbformat as nbf
        with open(paths["notebook"]) as f:
            nb = nbf.read(f, as_version=4)

        # Should have many cells (comprehensive report)
        assert len(nb.cells) >= 10

        # Should contain markdown and code cells
        cell_types = [cell.cell_type for cell in nb.cells]
        assert "markdown" in cell_types
        assert "code" in cell_types

    def test_research_report_without_dependencies(self, converged_result, tmp_path, monkeypatch):
        """Test graceful handling when dependencies unavailable."""
        # Mock import to fail
        import sys
        monkeypatch.setitem(sys.modules, "plotly", None)

        with pytest.raises(ImportError, match="notebook support"):
            converged_result.create_research_report(
                title="Test",
                problem_config={},
                output_dir=str(tmp_path)
            )
```

**Estimated Lines**: ~80 lines

#### 3. Documentation (Phase 2)
**File**: `examples/basic/solver_result_analysis_demo.py`

**Add Demo Function**:
```python
def demo_create_research_report():
    """Demonstrate comprehensive research report creation."""
    print("\n" + "=" * 60)
    print("Demo 6: Comprehensive Research Report")
    print("=" * 60)

    try:
        import plotly
    except ImportError:
        print("⚠️  Plotly not available - skipping research report demo")
        print("   Install with: pip install plotly")
        return

    # Create result
    result = create_sample_result("Research Solver", converged=True, iterations=100)

    # Create comprehensive report
    print("\nCreating comprehensive research report...")
    paths = result.create_research_report(
        title="MFG Research Analysis",
        problem_config={
            "sigma": 0.5,
            "T": 1.0,
            "Nx": 50,
            "Nt": 30
        },
        output_dir=str(OUTPUT_DIR),
        export_html=True
    )

    print(f"\n✅ Research report created:")
    print(f"   Notebook: {paths['notebook']}")
    if 'html' in paths:
        print(f"   HTML: {paths['html']}")

    print("\nResearch report includes:")
    print("  • Mathematical framework with LaTeX equations")
    print("  • Interactive Plotly visualizations")
    print("  • Comprehensive convergence analysis")
    print("  • Mass conservation tracking")
    print("  • Professional documentation")
```

**Update main()**:
```python
def main():
    """Run all demonstrations."""
    # ... existing demos ...
    demo_export_notebook()          # Phase 1
    demo_create_research_report()   # Phase 2
    # ...
```

## Timeline Estimate

| Phase | Task | Duration |
|:------|:-----|:---------|
| **Phase 1** | Implement `_export_notebook()` | 2 hours |
| | Add tests (4-5 tests) | 1 hour |
| | Update demo and docs | 30 min |
| | **Phase 1 Subtotal** | **3.5 hours** |
| **Phase 2** | Implement `create_research_report()` | 1.5 hours |
| | Add tests (4-5 tests) | 1 hour |
| | Update demo and docs | 30 min |
| | **Phase 2 Subtotal** | **3 hours** |
| | **Total (Both Phases)** | **6.5 hours** |

**Size Classification**:
- Phase 1: Small-Medium
- Phase 2: Small
- Combined: Medium

## Success Criteria

**Phase 1 Completion**:
- ✅ `export_summary(output_format="notebook")` creates valid .ipynb files
- ✅ Generated notebooks contain 6+ cells with proper structure
- ✅ All new tests passing (4-5 tests)
- ✅ Demo example works end-to-end
- ✅ Documentation updated
- ✅ Graceful handling when nbformat unavailable

**Phase 2 Completion**:
- ✅ `create_research_report()` method added to SolverResult
- ✅ Wraps MFGNotebookReporter correctly
- ✅ Generates comprehensive notebooks with Plotly visualizations
- ✅ All new tests passing (4-5 tests)
- ✅ Demo example works with plotly installed
- ✅ Graceful handling when dependencies unavailable
- ✅ HTML export works when enabled

**Overall Success**:
- ✅ Both lightweight and comprehensive options available
- ✅ Simple, consistent API
- ✅ No code duplication
- ✅ All tests passing
- ✅ Documentation complete

## Related Work

**Existing Issues**:
- Issue #127: Solver Result Analysis Tools ✅ (Completed)

**Related Features**:
- `MFGNotebookReporter` (`mfg_pde/utils/notebooks/reporting.py`)
- `analyze_convergence()`, `plot_convergence()`, `compare_to()` methods
- Markdown/LaTeX export infrastructure

## Future Enhancements (Beyond Phase 2)

Potential future extensions:

1. **Enhanced Visualization Templates**
   - Additional Plotly template styles (dark mode, minimal, etc.)
   - Custom color schemes for specific journals
   - Animated visualizations for presentations

2. **Batch Reporting**
   - Generate comparative reports from multiple SolverResult objects
   - Automated benchmarking notebooks
   - Parameter study visualizations

3. **Template Customization**
   - User-defined report sections
   - Custom cell insertion hooks
   - Report template library (research, presentation, teaching)

4. **Integration with Other Tools**
   - Export to LaTeX-compatible formats for papers
   - Integration with experiment tracking (MLflow, Weights & Biases)
   - Cloud notebook support (Colab, Kaggle)

**Estimated Effort**: 8-15 hours (future work based on demand)

---

**Document Status**: ✅ Complete - Updated for both phases
**Next Steps**: Update Issue #134, then proceed with implementation
**Owner**: Development team
**Phases**: Both Phase 1 and Phase 2 approved for implementation
