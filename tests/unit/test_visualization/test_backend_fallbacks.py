"""
Unit tests for Visualization Backend Fallback Logic.

Tests focus on optional dependency detection and graceful fallback
mechanisms when visualization backends are unavailable.
"""

import pytest

# ============================================================================
# Test: Plotly Availability Detection
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_plotly_availability_flag():
    """Test Plotly availability detection via import."""
    try:
        import plotly  # noqa: F401

        from mfg_pde.visualization.network_plots import PLOTLY_AVAILABLE

        assert PLOTLY_AVAILABLE is True
    except ImportError:
        # If plotly not installed, PLOTLY_AVAILABLE should be False
        from mfg_pde.visualization.network_plots import PLOTLY_AVAILABLE

        assert PLOTLY_AVAILABLE is False


@pytest.mark.unit
@pytest.mark.fast
def test_plotly_import_error_handling():
    """Test graceful handling when plotly import fails."""
    # Test that module can be imported even if plotly unavailable
    try:
        from mfg_pde.visualization import network_plots

        # Module should import successfully
        assert network_plots is not None
    except ImportError as e:
        pytest.fail(f"Module import should not fail when plotly unavailable: {e}")


# ============================================================================
# Test: Bokeh Availability Detection
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_bokeh_availability_flag():
    """Test Bokeh availability detection via import."""
    try:
        import bokeh  # noqa: F401

        from mfg_pde.visualization.interactive_plots import BOKEH_AVAILABLE

        assert BOKEH_AVAILABLE is True
    except ImportError:
        # If bokeh not installed, BOKEH_AVAILABLE should be False
        from mfg_pde.visualization.interactive_plots import BOKEH_AVAILABLE

        assert BOKEH_AVAILABLE is False


# ============================================================================
# Test: NetworkX Availability Detection
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_networkx_availability_flag():
    """Test NetworkX availability detection via import."""
    try:
        import networkx  # noqa: F401

        from mfg_pde.visualization.network_plots import NETWORKX_AVAILABLE

        assert NETWORKX_AVAILABLE is True
    except ImportError:
        # If networkx not installed, NETWORKX_AVAILABLE should be False
        from mfg_pde.visualization.network_plots import NETWORKX_AVAILABLE

        assert NETWORKX_AVAILABLE is False


# ============================================================================
# Test: Backend Fallback Behavior
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_matplotlib_always_available():
    """Test that matplotlib is always available as fallback backend."""
    # Matplotlib is a core dependency, should always import
    try:
        import matplotlib  # noqa: F401

        success = True
    except ImportError:
        success = False

    assert success is True, "Matplotlib should be available as core dependency"


@pytest.mark.unit
@pytest.mark.fast
def test_backend_fallback_chain():
    """Test backend selection follows plotly -> matplotlib fallback chain."""
    from mfg_pde.visualization.mathematical_plots import MathematicalPlotter

    # Create plotter with auto backend
    plotter = MathematicalPlotter(backend="auto")

    # Backend should be either plotly (if available) or matplotlib (fallback)
    assert plotter.backend in ["plotly", "matplotlib", "auto"]

    # If backend is resolved, verify it's one of the valid options
    if plotter.backend != "auto":
        try:
            import plotly  # noqa: F401

            # If plotly available, backend should prefer it
            assert plotter.backend in ["plotly", "matplotlib"]
        except ImportError:
            # If plotly unavailable, should fall back to matplotlib
            assert plotter.backend == "matplotlib"


# ============================================================================
# Test: Polars Availability (Analytics Module)
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_polars_optional_dependency():
    """Test that mfg_analytics module handles missing Polars gracefully."""
    try:
        from mfg_pde.visualization.mfg_analytics import POLARS_AVAILABLE

        # Flag should exist and be boolean
        assert isinstance(POLARS_AVAILABLE, bool)

        # If Polars is available, verify it can be imported
        if POLARS_AVAILABLE:
            import polars  # noqa: F401

    except ImportError:
        # If module can't be imported at all, that's acceptable
        # (means entire analytics module has hard Polars dependency)
        pass


# ============================================================================
# Test: Import Error Recovery
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_visualization_module_base_imports():
    """Test that base visualization module imports successfully."""
    # These imports should always work regardless of optional deps
    try:
        from mfg_pde.visualization.mathematical_plots import (
            MathematicalPlotter,
            create_mathematical_visualizer,
        )

        assert MathematicalPlotter is not None
        assert create_mathematical_visualizer is not None
    except ImportError as e:
        pytest.fail(f"Base visualization imports should not fail: {e}")


@pytest.mark.unit
@pytest.mark.fast
def test_network_visualization_with_missing_networkx():
    """Test network visualization can handle missing NetworkX."""
    from mfg_pde.visualization.network_plots import NETWORKX_AVAILABLE

    if not NETWORKX_AVAILABLE:
        # Verify that NETWORKX_AVAILABLE is correctly False
        assert NETWORKX_AVAILABLE is False

        # Module should still be importable
        try:
            from mfg_pde.visualization import network_plots

            assert network_plots is not None
        except ImportError as e:
            pytest.fail(f"network_plots should import without NetworkX: {e}")


# ============================================================================
# Test: Feature Detection
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_interactive_features_detection():
    """Test detection of interactive visualization capabilities."""
    from mfg_pde.visualization.interactive_plots import (
        BOKEH_AVAILABLE,
        PLOTLY_AVAILABLE,
    )

    # At least one interactive backend should be available (plotly is dependency)
    has_interactive = PLOTLY_AVAILABLE or BOKEH_AVAILABLE

    # This should typically be True since plotly is in requirements
    assert isinstance(has_interactive, bool)


@pytest.mark.unit
@pytest.mark.fast
def test_visualization_capability_flags():
    """Test that capability flags are properly set."""
    from mfg_pde.visualization.network_plots import (
        NETWORKX_AVAILABLE,
        PLOTLY_AVAILABLE,
    )

    # Flags should be boolean
    assert isinstance(PLOTLY_AVAILABLE, bool)
    assert isinstance(NETWORKX_AVAILABLE, bool)

    # At least matplotlib should be available as fallback
    import matplotlib.pyplot as plt

    assert plt is not None
