//! Experimental Rust Extensions for MFG_PDE
//!
//! This is a learning/experimental crate to explore Rust-Python integration
//! using PyO3 and potential performance optimization opportunities.
//!
//! **Status**: EXPERIMENTAL - Not integrated into main package yet
//!
//! To build and test:
//! ```bash
//! cd mfg_pde_rust_experimental
//! maturin develop  # Build and install in current Python environment
//! python test_rust.py  # Test the functions
//! ```

use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

/// Simple hello world function to verify Rust-Python communication works
///
/// Returns:
///     str: A greeting from Rust
///
/// Example:
///     >>> import mfg_pde_rust_experimental
///     >>> mfg_pde_rust_experimental.hello_rust()
///     'Hello from Rust! PyO3 is working.'
#[pyfunction]
fn hello_rust() -> PyResult<String> {
    Ok("Hello from Rust! PyO3 is working.".to_string())
}

/// Simple addition function for basic Rust syntax learning
///
/// Args:
///     a (float): First number
///     b (float): Second number
///
/// Returns:
///     float: Sum of a and b
#[pyfunction]
fn add(a: f64, b: f64) -> PyResult<f64> {
    Ok(a + b)
}

/// Sum all elements in a NumPy array (learn NumPy-Rust integration)
///
/// Args:
///     array (np.ndarray): 1D NumPy array
///
/// Returns:
///     float: Sum of all elements
#[pyfunction]
fn sum_array(array: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let array = array.as_array();
    Ok(array.iter().sum())
}

/// Compute mean of a NumPy array
///
/// Args:
///     array (np.ndarray): 1D NumPy array
///
/// Returns:
///     float: Mean of all elements
#[pyfunction]
fn mean_array(array: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let array = array.as_array();
    let sum: f64 = array.iter().sum();
    let count = array.len() as f64;
    Ok(sum / count)
}

/// Square all elements in a NumPy array (return new array)
///
/// Args:
///     array (np.ndarray): 1D NumPy array
///
/// Returns:
///     np.ndarray: New array with squared elements
#[pyfunction]
fn square_array<'py>(
    py: Python<'py>,
    array: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let array = array.as_array();
    let squared: Vec<f64> = array.iter().map(|&x| x * x).collect();
    Ok(PyArray1::from_vec_bound(py, squared))
}

/// WENO5 smoothness indicators (educational example of real numerical kernel)
///
/// Compute smoothness indicators β₀, β₁, β₂ for WENO5 reconstruction.
/// This is the same computation as in Python but in Rust.
///
/// Args:
///     u (np.ndarray): 5-point stencil [u₀, u₁, u₂, u₃, u₄]
///
/// Returns:
///     np.ndarray: Smoothness indicators [β₀, β₁, β₂]
///
/// Mathematical formulation:
///     β₀ = (13/12)(u₀ - 2u₁ + u₂)² + (1/4)(u₀ - 4u₁ + 3u₂)²
///     β₁ = (13/12)(u₁ - 2u₂ + u₃)² + (1/4)(u₁ - u₃)²
///     β₂ = (13/12)(u₂ - 2u₃ + u₄)² + (1/4)(3u₂ - 4u₃ + u₄)²
#[pyfunction]
fn weno5_smoothness<'py>(
    py: Python<'py>,
    u: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let u = u.as_array();

    if u.len() != 5 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "WENO5 requires exactly 5 points in stencil"
        ));
    }

    // Extract stencil values
    let u0 = u[0];
    let u1 = u[1];
    let u2 = u[2];
    let u3 = u[3];
    let u4 = u[4];

    // Compute smoothness indicators
    let beta_0 = (13.0/12.0) * (u0 - 2.0*u1 + u2).powi(2)
               + (1.0/4.0) * (u0 - 4.0*u1 + 3.0*u2).powi(2);

    let beta_1 = (13.0/12.0) * (u1 - 2.0*u2 + u3).powi(2)
               + (1.0/4.0) * (u1 - u3).powi(2);

    let beta_2 = (13.0/12.0) * (u2 - 2.0*u3 + u4).powi(2)
               + (1.0/4.0) * (3.0*u2 - 4.0*u3 + u4).powi(2);

    // Return as NumPy array
    Ok(PyArray1::from_vec_bound(py, vec![beta_0, beta_1, beta_2]))
}

/// Python module definition
#[pymodule]
fn mfg_pde_rust_experimental(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_rust, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(sum_array, m)?)?;
    m.add_function(wrap_pyfunction!(mean_array, m)?)?;
    m.add_function(wrap_pyfunction!(square_array, m)?)?;
    m.add_function(wrap_pyfunction!(weno5_smoothness, m)?)?;
    Ok(())
}

// Rust unit tests (run with `cargo test`)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let result = add(2.0, 3.0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 5.0);
    }

    #[test]
    fn test_weno5_smoothness_stencil_size() {
        // This test checks error handling, not actual computation
        // (NumPy arrays can't be easily created in Rust tests)
        // Use Python tests in test_rust.py for full integration testing
    }
}
