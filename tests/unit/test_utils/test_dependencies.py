"""
Unit tests for Optional Dependency Checking System.

Tests the dependency utilities that provide helpful error messages for
missing optional dependencies.
"""

import pytest

# ============================================================================
# Test: is_available() Function
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_is_available_core_dependency():
    """Test is_available() returns True for core dependencies."""
    from mfg_pde.utils.dependencies import is_available

    # NumPy is a core dependency, should always be available
    assert is_available("numpy") is True
    assert is_available("scipy") is True
    assert is_available("matplotlib") is True


@pytest.mark.unit
@pytest.mark.fast
def test_is_available_nonexistent_package():
    """Test is_available() returns False for nonexistent packages."""
    from mfg_pde.utils.dependencies import is_available

    # Package that definitely doesn't exist
    assert is_available("this_package_does_not_exist_xyz123") is False


@pytest.mark.unit
@pytest.mark.fast
def test_is_available_optional_dependency():
    """Test is_available() correctly detects optional dependencies."""
    from mfg_pde.utils.dependencies import is_available

    # Test optional dependencies - results depend on environment
    torch_available = is_available("torch")
    jax_available = is_available("jax")
    plotly_available = is_available("plotly")

    # Should return boolean values
    assert isinstance(torch_available, bool)
    assert isinstance(jax_available, bool)
    assert isinstance(plotly_available, bool)


# ============================================================================
# Test: check_dependency() Function
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_check_dependency_available():
    """Test check_dependency() returns True for available packages."""
    from mfg_pde.utils.dependencies import check_dependency

    # Core dependency should pass
    assert check_dependency("numpy") is True


@pytest.mark.unit
@pytest.mark.fast
def test_check_dependency_missing_known_package():
    """Test check_dependency() raises helpful error for known packages."""
    from mfg_pde.utils.dependencies import check_dependency

    # Try to check a known optional dependency that might not be installed
    # We'll use a package unlikely to be installed
    try:
        check_dependency("this_package_does_not_exist_xyz123")
        pytest.fail("Expected ImportError for missing package")
    except ImportError as e:
        error_msg = str(e)
        # Should contain helpful message
        assert "required" in error_msg.lower()
        assert "install" in error_msg.lower()


@pytest.mark.unit
@pytest.mark.fast
def test_check_dependency_with_feature_description():
    """Test check_dependency() includes feature in error message."""
    from mfg_pde.utils.dependencies import check_dependency

    try:
        check_dependency("nonexistent_package_xyz", feature="test feature")
        pytest.fail("Expected ImportError")
    except ImportError as e:
        error_msg = str(e)
        # Should include feature description
        assert "test feature" in error_msg


@pytest.mark.unit
@pytest.mark.fast
def test_check_dependency_known_package_error_format():
    """Test error message format for known optional dependencies."""
    from mfg_pde.utils.dependencies import DEPENDENCY_MAP, check_dependency

    # Pick a known package from DEPENDENCY_MAP
    known_packages = list(DEPENDENCY_MAP.keys())
    test_package = known_packages[0]  # e.g., 'torch'

    # If package is not available, check error message format
    from mfg_pde.utils.dependencies import is_available

    if not is_available(test_package):
        try:
            check_dependency(test_package)
            pytest.fail(f"Expected ImportError for {test_package}")
        except ImportError as e:
            error_msg = str(e)
            # Should contain structured error message
            assert "Used by:" in error_msg
            assert "Install options:" in error_msg
            assert "pip install" in error_msg


# ============================================================================
# Test: require_dependencies() Decorator
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_require_dependencies_decorator_success():
    """Test require_dependencies decorator allows execution when deps available."""
    from mfg_pde.utils.dependencies import require_dependencies

    @require_dependencies("numpy", "scipy")
    def test_function():
        return "success"

    # Should execute successfully
    result = test_function()
    assert result == "success"


@pytest.mark.unit
@pytest.mark.fast
def test_require_dependencies_decorator_failure():
    """Test require_dependencies decorator raises error for missing deps."""
    from mfg_pde.utils.dependencies import require_dependencies

    @require_dependencies("nonexistent_package_xyz")
    def test_function():
        return "should not reach here"

    # Should raise ImportError before executing function
    with pytest.raises(ImportError) as exc_info:
        test_function()

    assert "nonexistent_package_xyz" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.fast
def test_require_dependencies_with_feature():
    """Test require_dependencies decorator includes feature in error."""
    from mfg_pde.utils.dependencies import require_dependencies

    @require_dependencies("nonexistent_xyz", feature="test functionality")
    def test_function():
        return "should not reach here"

    with pytest.raises(ImportError) as exc_info:
        test_function()

    error_msg = str(exc_info.value)
    assert "test functionality" in error_msg


# ============================================================================
# Test: get_available_features() Function
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_get_available_features_structure():
    """Test get_available_features returns correct structure."""
    from mfg_pde.utils.dependencies import get_available_features

    features = get_available_features()

    # Should return dictionary
    assert isinstance(features, dict)

    # Should contain expected keys
    expected_features = ["pytorch", "jax", "gymnasium", "plotly", "networkx"]
    for feature in expected_features:
        assert feature in features

    # All values should be boolean
    for feature, available in features.items():
        assert isinstance(available, bool), f"{feature} availability should be boolean"


@pytest.mark.unit
@pytest.mark.fast
def test_get_available_features_consistency():
    """Test get_available_features is consistent with is_available."""
    from mfg_pde.utils.dependencies import get_available_features, is_available

    features = get_available_features()

    # PyTorch feature should match torch package
    assert features["pytorch"] == is_available("torch")

    # JAX feature should match jax package
    assert features["jax"] == is_available("jax")


# ============================================================================
# Test: Module-Level Availability Flags
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_module_level_flags_exist():
    """Test that module-level availability flags are defined."""
    from mfg_pde.utils import dependencies

    # These flags should exist
    assert hasattr(dependencies, "TORCH_AVAILABLE")
    assert hasattr(dependencies, "JAX_AVAILABLE")
    assert hasattr(dependencies, "GYMNASIUM_AVAILABLE")
    assert hasattr(dependencies, "PLOTLY_AVAILABLE")
    assert hasattr(dependencies, "NETWORKX_AVAILABLE")

    # Should all be boolean
    assert isinstance(dependencies.TORCH_AVAILABLE, bool)
    assert isinstance(dependencies.JAX_AVAILABLE, bool)
    assert isinstance(dependencies.GYMNASIUM_AVAILABLE, bool)


@pytest.mark.unit
@pytest.mark.fast
def test_module_level_flags_consistency():
    """Test module-level flags match is_available results."""
    from mfg_pde.utils import dependencies
    from mfg_pde.utils.dependencies import is_available

    assert is_available("torch") == dependencies.TORCH_AVAILABLE
    assert is_available("jax") == dependencies.JAX_AVAILABLE
    assert is_available("gymnasium") == dependencies.GYMNASIUM_AVAILABLE
    assert is_available("plotly") == dependencies.PLOTLY_AVAILABLE
    assert is_available("networkx") == dependencies.NETWORKX_AVAILABLE


# ============================================================================
# Test: show_optional_features() Function
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_show_optional_features_executes(capsys):
    """Test show_optional_features executes without error."""
    from mfg_pde.utils.dependencies import show_optional_features

    # Should execute without raising exception
    show_optional_features()

    # Capture output
    captured = capsys.readouterr()

    # Should produce some output
    assert len(captured.out) > 0
    assert "MFG_PDE Optional Features" in captured.out


@pytest.mark.unit
@pytest.mark.fast
def test_show_optional_features_format(capsys):
    """Test show_optional_features output format."""
    from mfg_pde.utils.dependencies import show_optional_features

    show_optional_features()
    captured = capsys.readouterr()

    # Should contain section headers
    assert "MFG_PDE Optional Features" in captured.out
    assert "Installation options:" in captured.out

    # Should contain feature names
    assert "pytorch" in captured.out or "PyTorch" in captured.out
    assert "jax" in captured.out or "JAX" in captured.out


# ============================================================================
# Test: Package-Level Function
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_package_level_show_features(capsys):
    """Test show_optional_features is accessible from package level."""
    import mfg_pde

    # Should be importable from package level
    assert hasattr(mfg_pde, "show_optional_features")

    # Should execute successfully
    mfg_pde.show_optional_features()

    captured = capsys.readouterr()
    assert len(captured.out) > 0
    assert "MFG_PDE Optional Features" in captured.out


# ============================================================================
# Test: DEPENDENCY_MAP Structure
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_dependency_map_structure():
    """Test DEPENDENCY_MAP has correct structure."""
    from mfg_pde.utils.dependencies import DEPENDENCY_MAP

    # Should be a dictionary
    assert isinstance(DEPENDENCY_MAP, dict)

    # Should contain key packages
    expected_packages = ["torch", "jax", "gymnasium", "plotly", "networkx"]
    for package in expected_packages:
        assert package in DEPENDENCY_MAP

    # Each entry should have required fields
    for _package, info in DEPENDENCY_MAP.items():
        assert "install_group" in info
        assert "install_cmd" in info
        assert "alternative" in info
        assert "used_by" in info

        # used_by should be a list
        assert isinstance(info["used_by"], list)
        assert len(info["used_by"]) > 0


@pytest.mark.unit
@pytest.mark.fast
def test_dependency_map_install_commands():
    """Test DEPENDENCY_MAP contains valid install commands."""
    from mfg_pde.utils.dependencies import DEPENDENCY_MAP

    for package, info in DEPENDENCY_MAP.items():
        # Install command should reference the package
        assert "pip install" in info["install_cmd"]

        # Alternative should also be pip install
        assert "pip install" in info["alternative"]
        assert package in info["alternative"]
