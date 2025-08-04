"""
Polars integration for efficient data manipulation in MFG_PDE.

This module provides high-performance data manipulation capabilities using Polars,
significantly outperforming pandas for large-scale MFG computations and analysis.

Features:
- Fast parameter sweep result processing
- Efficient time series analysis
- High-performance data aggregation
- Memory-efficient large dataset handling
- Integration with existing NumPy arrays
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

    # Create mock pl module for type hints
    class MockPolars:
        DataFrame = type(None)
        Series = type(None)

    pl = MockPolars()

logger = logging.getLogger(__name__)


class MFGDataFrame:
    """
    High-performance MFG data container using Polars.

    Provides efficient storage and manipulation of MFG simulation results,
    parameter sweeps, and convergence data.
    """

    def __init__(
        self,
        data: Optional[Union[pl.DataFrame, Dict[str, Any], List[Dict[str, Any]], np.ndarray]] = None,
    ):
        """
        Initialize MFG DataFrame.

        Args:
            data: Initial data (Polars DataFrame, dict, or NumPy array)
        """
        if not POLARS_AVAILABLE:
            raise ImportError("Polars not available. Install with: pip install polars")

        if data is None:
            self.df = pl.DataFrame()
        elif isinstance(data, pl.DataFrame):
            self.df = data
        elif isinstance(data, dict):
            self.df = pl.DataFrame(data)
        elif isinstance(data, list):
            # Handle list of dictionaries
            self.df = pl.DataFrame(data)
        elif isinstance(data, np.ndarray):
            # Convert NumPy array to DataFrame with auto-generated column names
            if data.ndim == 1:
                self.df = pl.DataFrame({"values": data})
            elif data.ndim == 2:
                columns = [f"col_{i}" for i in range(data.shape[1])]
                self.df = pl.DataFrame(data, schema=columns)
            else:
                raise ValueError("Only 1D and 2D NumPy arrays are supported")
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    @property
    def shape(self) -> Tuple[int, int]:
        """Get DataFrame shape."""
        return self.df.shape

    @property
    def columns(self) -> List[str]:
        """Get column names."""
        return self.df.columns

    def __len__(self) -> int:
        """Get number of rows."""
        return len(self.df)

    def __getitem__(self, key: str) -> pl.Series:
        """Get column by name."""
        return self.df[key]

    def to_numpy(self, column: Optional[str] = None) -> np.ndarray:
        """
        Convert to NumPy array.

        Args:
            column: Specific column to convert (if None, convert all)

        Returns:
            NumPy array
        """
        if column:
            return self.df[column].to_numpy()
        return self.df.to_numpy()

    def to_dict(self) -> Dict[str, List[Any]]:
        """Convert to dictionary."""
        return self.df.to_dict()

    def head(self, n: int = 5) -> pl.DataFrame:
        """Get first n rows."""
        return self.df.head(n)

    def describe(self) -> pl.DataFrame:
        """Generate descriptive statistics."""
        return self.df.describe()


class MFGParameterSweepAnalyzer:
    """
    High-performance analyzer for MFG parameter sweep results using Polars.
    """

    def __init__(self):
        """Initialize parameter sweep analyzer."""
        if not POLARS_AVAILABLE:
            raise ImportError("Polars not available. Install with: pip install polars")

    def create_sweep_dataframe(self, results: List[Dict[str, Any]]) -> MFGDataFrame:
        """
        Create DataFrame from parameter sweep results.

        Args:
            results: List of result dictionaries from parameter sweeps

        Returns:
            MFGDataFrame containing organized results
        """
        # Flatten nested result dictionaries
        flattened_results = []

        for i, result in enumerate(results):
            flattened = {"run_id": i}

            # Flatten parameters
            if "parameters" in result:
                for key, value in result["parameters"].items():
                    flattened[f"param_{key}"] = value

            # Flatten metrics
            for key, value in result.items():
                if key not in ["parameters", "x_grid", "final_density", "M"]:
                    if isinstance(value, (int, float, str, bool)):
                        flattened[f"metric_{key}"] = value

            # Handle array data separately
            if "final_density" in result:
                density = result["final_density"]
                if isinstance(density, np.ndarray):
                    # Store statistical summaries of density
                    flattened["density_mean"] = np.mean(density)
                    flattened["density_std"] = np.std(density)
                    flattened["density_min"] = np.min(density)
                    flattened["density_max"] = np.max(density)
                    flattened["density_range"] = np.max(density) - np.min(density)

            flattened_results.append(flattened)

        return MFGDataFrame(flattened_results)

    def analyze_parameter_effects(
        self, df: MFGDataFrame, parameter_cols: List[str], metric_cols: List[str]
    ) -> pl.DataFrame:
        """
        Analyze effects of parameters on metrics.

        Args:
            df: MFGDataFrame containing sweep results
            parameter_cols: Parameter column names
            metric_cols: Metric column names

        Returns:
            Analysis results DataFrame
        """
        results = []

        for param in parameter_cols:
            for metric in metric_cols:
                # Group by parameter and compute statistics
                grouped = df.df.group_by(param).agg(
                    [
                        pl.col(metric).mean().alias(f"{metric}_mean"),
                        pl.col(metric).std().alias(f"{metric}_std"),
                        pl.col(metric).min().alias(f"{metric}_min"),
                        pl.col(metric).max().alias(f"{metric}_max"),
                        pl.col(metric).count().alias(f"{metric}_count"),
                    ]
                )

                # Add parameter and metric info
                for row in grouped.iter_rows(named=True):
                    result = {
                        "parameter": param,
                        "metric": metric,
                        "param_value": row[param],
                        "mean": row[f"{metric}_mean"],
                        "std": row[f"{metric}_std"],
                        "min": row[f"{metric}_min"],
                        "max": row[f"{metric}_max"],
                        "count": row[f"{metric}_count"],
                    }
                    results.append(result)

        return pl.DataFrame(results)

    def compute_correlation_matrix(self, df: MFGDataFrame, columns: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Compute correlation matrix between columns.

        Args:
            df: MFGDataFrame
            columns: Columns to include (if None, use all numeric columns)

        Returns:
            Correlation matrix DataFrame
        """
        if columns is None:
            # Get numeric columns
            numeric_cols = []
            for col in df.columns:
                if df.df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                    numeric_cols.append(col)
            columns = numeric_cols

        # Compute correlations
        correlations = []
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i <= j:  # Only compute upper triangle
                    corr = df.df.select([pl.corr(col1, col2).alias("correlation")]).item()

                    correlations.append({"column1": col1, "column2": col2, "correlation": corr})

        return pl.DataFrame(correlations)

    def find_optimal_parameters(self, df: MFGDataFrame, objective_column: str, minimize: bool = True) -> Dict[str, Any]:
        """
        Find optimal parameter combination based on objective.

        Args:
            df: MFGDataFrame containing sweep results
            objective_column: Column to optimize
            minimize: Whether to minimize (True) or maximize (False)

        Returns:
            Dictionary with optimal parameters and objective value
        """
        if minimize:
            optimal_row = df.df.filter(pl.col(objective_column) == pl.col(objective_column).min()).limit(1)
        else:
            optimal_row = df.df.filter(pl.col(objective_column) == pl.col(objective_column).max()).limit(1)

        if len(optimal_row) == 0:
            raise ValueError(f"No valid values found in column {objective_column}")

        return optimal_row.to_dicts()[0]


class MFGTimeSeriesAnalyzer:
    """
    High-performance time series analysis for MFG convergence data.
    """

    def __init__(self):
        """Initialize time series analyzer."""
        if not POLARS_AVAILABLE:
            raise ImportError("Polars not available. Install with: pip install polars")

    def create_convergence_dataframe(self, convergence_history: List[Dict[str, Any]]) -> MFGDataFrame:
        """
        Create DataFrame from convergence history.

        Args:
            convergence_history: List of convergence data per iteration

        Returns:
            MFGDataFrame with convergence time series
        """
        # Ensure all entries have iteration numbers
        for i, entry in enumerate(convergence_history):
            if "iteration" not in entry:
                entry["iteration"] = i

        return MFGDataFrame(convergence_history)

    def analyze_convergence_rate(self, df: MFGDataFrame, error_column: str = "error") -> Dict[str, float]:
        """
        Analyze convergence rate from error time series.

        Args:
            df: MFGDataFrame with convergence data
            error_column: Column containing error values

        Returns:
            Dictionary with convergence statistics
        """
        # Compute convergence metrics
        error_series = df.df[error_column]

        # Linear convergence rate (log scale)
        log_errors = error_series.log()
        iterations = pl.Series("iteration", range(len(error_series)))

        # Fit linear regression to log(error) vs iteration
        # Using Polars operations for efficiency
        n = len(error_series)
        sum_x = iterations.sum()
        sum_y = log_errors.sum()
        sum_xy = (iterations * log_errors).sum()
        sum_x2 = (iterations * iterations).sum()

        # Linear regression coefficients
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n

        # Convergence rate (negative slope indicates convergence)
        convergence_rate = -slope

        # R-squared
        mean_y = sum_y / n
        ss_tot = ((log_errors - mean_y) ** 2).sum()
        ss_res = ((log_errors - (slope * iterations + intercept)) ** 2).sum()
        r_squared = 1 - (ss_res / ss_tot)

        return {
            "convergence_rate": convergence_rate,
            "linear_fit_slope": slope,
            "linear_fit_intercept": intercept,
            "r_squared": r_squared,
            "initial_error": error_series.first(),
            "final_error": error_series.last(),
            "error_reduction_factor": error_series.first() / error_series.last(),
            "iterations_to_convergence": len(error_series),
        }

    def detect_convergence_plateau(
        self,
        df: MFGDataFrame,
        error_column: str = "error",
        window_size: int = 10,
        plateau_threshold: float = 1e-8,
    ) -> Optional[int]:
        """
        Detect when convergence reaches a plateau.

        Args:
            df: MFGDataFrame with convergence data
            error_column: Column containing error values
            window_size: Size of moving window for plateau detection
            plateau_threshold: Threshold for plateau detection

        Returns:
            Iteration number where plateau starts, or None
        """
        error_series = df.df[error_column]

        if len(error_series) < window_size:
            return None

        # Compute moving window standard deviation
        moving_std = error_series.rolling_std(window_size)

        # Find first point where std deviation is below threshold
        plateau_mask = moving_std < plateau_threshold
        plateau_indices = plateau_mask.arg_true()

        if len(plateau_indices) > 0:
            return plateau_indices.first()

        return None


class MFGDataExporter:
    """
    Efficient data export capabilities using Polars.
    """

    def __init__(self):
        """Initialize data exporter."""
        if not POLARS_AVAILABLE:
            raise ImportError("Polars not available. Install with: pip install polars")

    def export_to_parquet(self, df: MFGDataFrame, filepath: Union[str, Path]) -> None:
        """
        Export DataFrame to Parquet format (highly efficient).

        Args:
            df: MFGDataFrame to export
            filepath: Output file path
        """
        df.df.write_parquet(filepath)
        logger.info(f"Data exported to Parquet: {filepath}")

    def export_to_csv(self, df: MFGDataFrame, filepath: Union[str, Path]) -> None:
        """
        Export DataFrame to CSV format.

        Args:
            df: MFGDataFrame to export
            filepath: Output file path
        """
        df.df.write_csv(filepath)
        logger.info(f"Data exported to CSV: {filepath}")

    def export_to_json(self, df: MFGDataFrame, filepath: Union[str, Path]) -> None:
        """
        Export DataFrame to JSON format.

        Args:
            df: MFGDataFrame to export
            filepath: Output file path
        """
        # Convert to JSON via dictionary with proper serialization
        import json

        data = df.to_dict()

        # Convert any non-serializable values to lists
        serializable_data = {}
        for key, value in data.items():
            if hasattr(value, "to_list"):  # Polars Series
                serializable_data[key] = value.to_list()
            elif isinstance(value, list):
                serializable_data[key] = value
            else:
                serializable_data[key] = str(value)

        with open(filepath, "w") as f:
            json.dump(serializable_data, f, indent=2)
        logger.info(f"Data exported to JSON: {filepath}")

    def load_from_parquet(self, filepath: Union[str, Path]) -> MFGDataFrame:
        """
        Load DataFrame from Parquet format.

        Args:
            filepath: Input file path

        Returns:
            MFGDataFrame
        """
        df = pl.read_parquet(filepath)
        logger.info(f"Data loaded from Parquet: {filepath}")
        return MFGDataFrame(df)

    def load_from_csv(self, filepath: Union[str, Path]) -> MFGDataFrame:
        """
        Load DataFrame from CSV format.

        Args:
            filepath: Input file path

        Returns:
            MFGDataFrame
        """
        df = pl.read_csv(filepath)
        logger.info(f"Data loaded from CSV: {filepath}")
        return MFGDataFrame(df)


def create_mfg_dataframe(data: Optional[Union[pl.DataFrame, Dict[str, Any], np.ndarray]] = None) -> MFGDataFrame:
    """
    Convenience function to create MFGDataFrame.

    Args:
        data: Initial data

    Returns:
        MFGDataFrame instance
    """
    return MFGDataFrame(data)


def create_parameter_sweep_analyzer() -> MFGParameterSweepAnalyzer:
    """
    Convenience function to create parameter sweep analyzer.

    Returns:
        MFGParameterSweepAnalyzer instance
    """
    return MFGParameterSweepAnalyzer()


def create_time_series_analyzer() -> MFGTimeSeriesAnalyzer:
    """
    Convenience function to create time series analyzer.

    Returns:
        MFGTimeSeriesAnalyzer instance
    """
    return MFGTimeSeriesAnalyzer()


def create_data_exporter() -> MFGDataExporter:
    """
    Convenience function to create data exporter.

    Returns:
        MFGDataExporter instance
    """
    return MFGDataExporter()


# Performance utilities
def benchmark_polars_vs_pandas(data_size: int = 100000) -> Dict[str, float]:
    """
    Benchmark Polars performance against pandas.

    Args:
        data_size: Size of test dataset

    Returns:
        Dictionary with timing results
    """
    if not POLARS_AVAILABLE:
        return {"error": "Polars not available"}

    import time

    # Generate test data
    np.random.seed(42)
    data = {
        "x": np.random.randn(data_size),
        "y": np.random.randn(data_size),
        "group": np.random.choice(["A", "B", "C"], data_size),
    }

    results = {}

    # Polars performance
    start_time = time.time()
    pl_df = pl.DataFrame(data)
    pl_result = pl_df.group_by("group").agg(
        [
            pl.col("x").mean().alias("x_mean"),
            pl.col("y").sum().alias("y_sum"),
            pl.col("x").std().alias("x_std"),
        ]
    )
    results["polars_time"] = time.time() - start_time

    # Try pandas comparison if available
    try:
        import pandas as pd

        start_time = time.time()
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.groupby("group").agg({"x": ["mean", "std"], "y": "sum"})
        results["pandas_time"] = time.time() - start_time
        results["speedup_factor"] = results["pandas_time"] / results["polars_time"]
    except ImportError:
        results["pandas_time"] = None
        results["speedup_factor"] = None

    results["data_size"] = data_size
    results["polars_result_shape"] = pl_result.shape

    return results
