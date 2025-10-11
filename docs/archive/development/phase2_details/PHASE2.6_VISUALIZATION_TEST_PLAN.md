# Phase 2.6: Visualization Logic Test Implementation Plan

**Date**: 2025-10-10
**Branch**: `test/phase2-coverage-expansion`
**Issue**: [#124 - Expand test coverage from 37% to 50%+](https://github.com/derrring/MFG_PDE/issues/124)
**Phase**: 2.6 (Visualization Logic Tests)

## Overview

Implement Phase 4 of Issue #124 by creating comprehensive tests for visualization module **data preparation and validation logic**, not UI rendering. Focus on testable computational logic within visualization methods.

## Current State

### Existing Coverage
- **Current**: 12-36% coverage across visualization modules
- **Existing tests**: `tests/unit/test_visualization_modules.py` (157 lines, 3 integration tests)
- **Module sizes**:
  - `interactive_plots.py`: 1,125 lines (largest, most complex)
  - `network_plots.py`: 776 lines (network-specific visualizations)
  - `enhanced_network_plots.py`: 624 lines
  - `multidim_viz.py`: 592 lines
  - `mfg_analytics.py`: 585 lines
  - `mathematical_plots.py`: 424 lines
  - `legacy_plotting.py`: 293 lines

### Testable Logic Categories

**1. Data Validation & Preprocessing**
- Input array shape validation
- Coordinate grid consistency checks
- Value range normalization
- Missing data handling

**2. Backend Selection & Fallback**
- Backend availability detection
- Graceful fallback mechanisms
- LaTeX support detection

**3. Coordinate Transformations**
- Grid coordinate extraction
- Network position calculations
- Edge coordinate computation
- Spatial scaling and normalization

**4. Network Data Extraction**
- Adjacency matrix processing
- Node position extraction
- Edge list generation
- Graph property calculation

**5. Statistical Computations**
- Network statistics calculation
- Convergence metrics
- Error norms and residuals
- Mass conservation checks

## Implementation Plan

### Test File Structure

```
tests/unit/test_visualization/
├── __init__.py
├── test_network_plots.py          # Network data extraction tests
├── test_mathematical_plots.py     # Backend selection & validation tests
├── test_mfg_analytics.py          # Analytics computation tests
├── test_coordinate_transforms.py  # Coordinate transformation tests
└── test_backend_fallbacks.py      # Backend availability tests
```

### Priority 1: Network Data Extraction Tests

**File**: `tests/unit/test_visualization/test_network_plots.py`

**Tests** (15 tests):

```python
@pytest.mark.unit
@pytest.mark.fast
def test_network_visualizer_initialization():
    """Test NetworkMFGVisualizer initialization with network data."""
    # Test with valid network_data
    # Test with problem containing network_data
    # Test error when neither provided

@pytest.mark.unit
@pytest.mark.fast
def test_network_properties_extraction():
    """Test extraction of network properties from data."""
    # Verify num_nodes, num_edges extracted correctly
    # Verify adjacency_matrix accessible
    # Verify node_positions accessible

@pytest.mark.unit
@pytest.mark.fast
def test_edge_coordinate_extraction():
    """Test extraction of edge coordinates from adjacency matrix."""
    # Test lines 142-149 in network_plots.py
    # Create small test network (3 nodes)
    # Verify edge coordinate lists generated correctly

@pytest.mark.unit
@pytest.mark.fast
def test_edge_extraction_with_no_edges():
    """Test edge extraction handles empty adjacency matrix."""
    # Test with zero adjacency matrix
    # Should return empty edge coordinate lists

@pytest.mark.unit
@pytest.mark.fast
def test_node_value_normalization():
    """Test node value normalization for coloring."""
    # Test value range mapping to colormap
    # Test handling of NaN values
    # Test handling of infinite values

@pytest.mark.unit
@pytest.mark.fast
def test_edge_value_normalization():
    """Test edge value normalization for coloring."""
    # Similar to node value tests but for edges

@pytest.mark.unit
def test_network_statistics_computation():
    """Test network statistics calculation."""
    # Test degree distribution calculation
    # Test clustering coefficient calculation
    # Test path length statistics

@pytest.mark.unit
@pytest.mark.fast
def test_node_position_validation():
    """Test validation of node position arrays."""
    # Test correct shape (num_nodes, 2)
    # Test handling of missing positions (should auto-generate)
    # Test coordinate range validation

@pytest.mark.unit
@pytest.mark.fast
def test_adjacency_matrix_validation():
    """Test adjacency matrix validation."""
    # Test square matrix requirement
    # Test non-negative weights
    # Test symmetry for undirected graphs

@pytest.mark.unit
@pytest.mark.fast
def test_density_array_shape_validation():
    """Test density array shape matches network."""
    # Test (num_nodes, num_timesteps) shape
    # Test error on shape mismatch

@pytest.mark.unit
@pytest.mark.fast
def test_value_function_array_validation():
    """Test value function array validation."""
    # Similar shape validation as density

@pytest.mark.unit
@pytest.mark.fast
def test_network_data_consistency_check():
    """Test consistency between adjacency matrix and positions."""
    # Number of nodes from matrix matches positions length
    # Edge indices within valid range

@pytest.mark.unit
@pytest.mark.fast
def test_default_visualization_parameters():
    """Test default parameter values set correctly."""
    # Verify default_node_size, default_edge_width, colorscale

@pytest.mark.unit
@pytest.mark.fast
def test_parameter_scaling_calculations():
    """Test node size and edge width scaling."""
    # Test node_size_scale multiplier
    # Test edge_width_scale multiplier

@pytest.mark.unit
@pytest.mark.fast
def test_colorscale_validation():
    """Test colorscale string validation."""
    # Test valid colorscale names
    # Test invalid colorscale handling
```

### Priority 2: Mathematical Plots & Backend Tests

**File**: `tests/unit/test_visualization/test_mathematical_plots.py`

**Tests** (12 tests):

```python
@pytest.mark.unit
@pytest.mark.fast
def test_backend_detection():
    """Test automatic backend detection."""
    # Test _validate_backend() method
    # Test "auto" selects plotly if available, else matplotlib

@pytest.mark.unit
@pytest.mark.fast
def test_backend_explicit_selection():
    """Test explicit backend selection."""
    # Test backend="plotly" when available
    # Test backend="matplotlib" always works
    # Test backend="invalid" raises error

@pytest.mark.unit
@pytest.mark.fast
def test_latex_setup_detection():
    """Test LaTeX availability detection."""
    # Test _setup_latex() method
    # Test graceful fallback if LaTeX unavailable

@pytest.mark.unit
@pytest.mark.fast
def test_input_array_validation():
    """Test input array shape and type validation."""
    # Test 1D array requirement for x, y in plot_mathematical_function
    # Test same length requirement
    # Test type conversion (list to array)

@pytest.mark.unit
@pytest.mark.fast
def test_grid_consistency_validation():
    """Test x_grid and t_grid consistency checks."""
    # Test density shape matches (len(x_grid), len(t_grid))
    # Test error on mismatch

@pytest.mark.unit
@pytest.mark.fast
def test_coordinate_range_validation():
    """Test coordinate range validation."""
    # Test monotonic increasing requirement
    # Test finite value requirement

@pytest.mark.unit
@pytest.mark.fast
def test_density_value_validation():
    """Test density array value validation."""
    # Test non-negative density requirement
    # Test finite value requirement
    # Test NaN handling

@pytest.mark.unit
@pytest.mark.fast
def test_gradient_computation_validation():
    """Test gradient array validation for HJB plots."""
    # Test gradient shape matches value function
    # Test finite gradient values

@pytest.mark.unit
@pytest.mark.fast
def test_phase_portrait_vector_validation():
    """Test vector field validation for phase portraits."""
    # Test u_field, v_field same shape as x, y grids
    # Test finite vector values

@pytest.mark.unit
@pytest.mark.fast
def test_title_and_label_sanitization():
    """Test string parameter sanitization."""
    # Test LaTeX special character handling
    # Test empty string handling

@pytest.mark.unit
@pytest.mark.fast
def test_save_path_validation():
    """Test save path validation and directory creation."""
    # Test valid path accepted
    # Test directory created if doesn't exist
    # Test invalid path handling

@pytest.mark.unit
@pytest.mark.fast
def test_backend_fallback_mechanism():
    """Test graceful fallback when preferred backend unavailable."""
    # Mock plotly unavailable
    # Verify matplotlib used instead
```

### Priority 3: MFG Analytics Tests

**File**: `tests/unit/test_visualization/test_mfg_analytics.py`

**Tests** (10 tests):

```python
@pytest.mark.unit
@pytest.mark.fast
def test_analytics_engine_initialization():
    """Test MFGAnalyticsEngine initialization."""
    # Test with default parameters
    # Test output_dir creation
    # Test prefer_plotly flag

@pytest.mark.unit
@pytest.mark.fast
def test_component_initialization():
    """Test analytics component initialization."""
    # Test viz_manager initialized when available
    # Test Polars components initialized when available

@pytest.mark.unit
@pytest.mark.fast
def test_convergence_metric_computation():
    """Test convergence metric calculation."""
    # Test L2 norm computation
    # Test relative error computation
    # Test residual computation

@pytest.mark.unit
@pytest.mark.fast
def test_mass_conservation_check():
    """Test mass conservation validation."""
    # Test density integration
    # Test conservation error computation
    # Test tolerance checking

@pytest.mark.unit
@pytest.mark.fast
def test_energy_computation():
    """Test energy functional calculation."""
    # Test kinetic energy term
    # Test potential energy term
    # Test total energy

@pytest.mark.unit
@pytest.mark.fast
def test_network_statistics():
    """Test network statistics computation."""
    # Test degree distribution
    # Test centrality measures
    # Test clustering coefficients

@pytest.mark.unit
@pytest.mark.fast
def test_time_series_data_extraction():
    """Test time series data extraction from solutions."""
    # Test density time series
    # Test value function time series
    # Test aggregate statistics time series

@pytest.mark.unit
@pytest.mark.fast
def test_parameter_sweep_result_aggregation():
    """Test parameter sweep result aggregation."""
    # Test result collection
    # Test statistical summary computation
    # Test optimal parameter identification

@pytest.mark.unit
@pytest.mark.fast
def test_output_directory_management():
    """Test output directory creation and management."""
    # Test directory creation
    # Test subdirectory organization
    # Test file path generation

@pytest.mark.unit
@pytest.mark.fast
def test_graceful_fallback_when_polars_unavailable():
    """Test analytics works without Polars."""
    # Mock POLARS_AVAILABLE = False
    # Verify basic analytics still function
```

### Priority 4: Coordinate Transformation Tests

**File**: `tests/unit/test_visualization/test_coordinate_transforms.py`

**Tests** (8 tests):

```python
@pytest.mark.unit
@pytest.mark.fast
def test_meshgrid_extraction():
    """Test extraction of X, Y from 1D grids."""
    # Test np.meshgrid usage patterns
    # Verify shape consistency

@pytest.mark.unit
@pytest.mark.fast
def test_coordinate_scaling():
    """Test coordinate scaling for visualization."""
    # Test normalization to [0, 1]
    # Test scaling to specific range

@pytest.mark.unit
@pytest.mark.fast
def test_network_layout_computation():
    """Test network layout algorithm output."""
    # Test spring layout coordinates
    # Test circular layout coordinates
    # Test hierarchical layout coordinates

@pytest.mark.unit
@pytest.mark.fast
def test_spatial_index_mapping():
    """Test mapping between spatial coordinates and array indices."""
    # Test coordinate to index conversion
    # Test index to coordinate conversion

@pytest.mark.unit
@pytest.mark.fast
def test_edge_midpoint_calculation():
    """Test edge midpoint computation for labels."""
    # Test midpoint formula
    # Test handling of curved edges

@pytest.mark.unit
@pytest.mark.fast
def test_bounding_box_computation():
    """Test bounding box calculation for plots."""
    # Test min/max coordinate extraction
    # Test margin addition

@pytest.mark.unit
@pytest.mark.fast
def test_aspect_ratio_calculation():
    """Test aspect ratio computation from data."""
    # Test x/y range ratio
    # Test equal aspect enforcement

@pytest.mark.unit
@pytest.mark.fast
def test_coordinate_transformation_chain():
    """Test chained coordinate transformations."""
    # Test data coords → normalized → plot coords
    # Test inverse transformations
```

### Priority 5: Backend Fallback Tests

**File**: `tests/unit/test_visualization/test_backend_fallbacks.py`

**Tests** (6 tests):

```python
@pytest.mark.unit
@pytest.mark.fast
def test_plotly_availability_check():
    """Test Plotly availability detection."""
    # Test PLOTLY_AVAILABLE flag
    # Test import error handling

@pytest.mark.unit
@pytest.mark.fast
def test_bokeh_availability_check():
    """Test Bokeh availability detection."""
    # Test BOKEH_AVAILABLE flag
    # Test import error handling

@pytest.mark.unit
@pytest.mark.fast
def test_networkx_availability_check():
    """Test NetworkX availability detection."""
    # Test NETWORKX_AVAILABLE flag
    # Test graceful degradation

@pytest.mark.unit
@pytest.mark.fast
def test_interactive_to_static_fallback():
    """Test fallback from interactive to static plots."""
    # Mock plotly unavailable
    # Verify matplotlib used instead

@pytest.mark.unit
@pytest.mark.fast
def test_latex_fallback():
    """Test fallback when LaTeX unavailable."""
    # Mock LaTeX unavailable
    # Verify plain text labels used

@pytest.mark.unit
@pytest.mark.fast
def test_polars_fallback():
    """Test fallback when Polars unavailable."""
    # Mock POLARS_AVAILABLE = False
    # Verify basic functionality preserved
```

## Implementation Strategy

### Phase 1: Setup and Structure (30 minutes)
1. Create test directory: `tests/unit/test_visualization/`
2. Create `__init__.py`
3. Set up test fixtures for common data structures
4. Create helper functions for test network/grid generation

### Phase 2: Network Tests (2 hours)
1. Implement `test_network_plots.py` (15 tests)
2. Focus on data extraction logic
3. Use small test networks (3-5 nodes)
4. Run tests incrementally

### Phase 3: Mathematical & Backend Tests (2 hours)
1. Implement `test_mathematical_plots.py` (12 tests)
2. Test backend selection without rendering
3. Test validation logic
4. Mock backend availability as needed

### Phase 4: Analytics & Transform Tests (1.5 hours)
1. Implement `test_mfg_analytics.py` (10 tests)
2. Implement `test_coordinate_transforms.py` (8 tests)
3. Test computational logic
4. Mock Polars if needed

### Phase 5: Backend Fallback Tests (30 minutes)
1. Implement `test_backend_fallbacks.py` (6 tests)
2. Test import error handling
3. Test graceful degradation

### Phase 6: Coverage Analysis (30 minutes)
1. Run pytest-cov on visualization modules
2. Identify remaining gaps
3. Add targeted tests for uncovered lines
4. Document coverage improvement

## Test Data Fixtures

Create shared fixtures in `tests/unit/test_visualization/conftest.py`:

```python
@pytest.fixture
def small_network_data():
    """Create small test network (3 nodes)."""
    adjacency_matrix = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    node_positions = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 1.0]
    ])
    # Return NetworkData instance

@pytest.fixture
def test_density_1d():
    """Create 1D test density."""
    x_grid = np.linspace(0, 1, 10)
    density = np.exp(-10 * (x_grid - 0.5)**2)
    density = density / np.trapz(density, x_grid)
    return x_grid, density

@pytest.fixture
def test_density_2d():
    """Create 2D test density (x, t)."""
    x_grid = np.linspace(0, 1, 10)
    t_grid = np.linspace(0, 1, 5)
    X, T = np.meshgrid(x_grid, t_grid, indexing='ij')
    density = np.exp(-10 * (X - 0.5)**2) * np.exp(-2 * T)
    return x_grid, t_grid, density

@pytest.fixture
def mock_mfg_problem():
    """Create minimal MFG problem for testing."""
    # Return problem with network_data
```

## Success Metrics

### Coverage Targets
- **Network plots**: 0% → 30% (+30%)
- **Mathematical plots**: 36% → 50% (+14%)
- **Analytics**: 12% → 35% (+23%)
- **Overall visualization**: 12-36% → 35-40% (+~10% average)

### Contribution to Issue #124
- **Phase 1 (Workflow)**: ✅ Complete (+12% coverage)
- **Phase 2-3 (Decorators/Progress)**: ✅ Complete (+4% coverage)
- **Phase 4 (Visualization)**: 🎯 Target (+4-5% overall coverage)

**Expected Final**: 53% → 57-58% overall coverage

### Quality Metrics
- All tests pass locally and in CI
- Test execution time < 10 seconds (fast unit tests)
- No matplotlib display windows opened (use Agg backend)
- No file system pollution (use temp directories)

## Key Testing Principles

**✅ DO Test:**
- Data validation logic
- Coordinate extraction and transformation
- Backend selection and fallback
- Statistical computations
- Error handling and edge cases
- Input sanitization

**❌ DON'T Test:**
- Actual plot rendering (visual output)
- Interactive widget behavior
- Animation playback
- File I/O (unless testing path validation)
- External library behavior (NetworkX layouts, Plotly rendering)

## Timeline

- **Day 1** (3 hours): Phases 1-2 (Setup + Network tests)
- **Day 2** (3 hours): Phase 3-4 (Mathematical + Analytics tests)
- **Day 3** (1.5 hours): Phase 5-6 (Fallback + Coverage)

**Total Effort**: ~7.5 hours over 3 sessions

## Related Documentation

- Issue #124: Test Coverage Expansion Plan
- Phase 2.5 Summary: Workflow Test Implementation
- Repository Standards: `CLAUDE.md` test section

---

**Status**: Ready to implement
**Next Step**: Create test directory structure and implement Priority 1 tests
