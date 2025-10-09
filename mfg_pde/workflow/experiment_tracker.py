"""
Experiment tracking and management for MFG_PDE workflow system.

This module provides comprehensive experiment lifecycle management,
including experiment creation, execution tracking, result storage,
and comparative analysis capabilities.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .workflow_manager import Workflow


class ExperimentStatus(Enum):
    """Experiment execution status."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentMetadata:
    """Metadata for experiment tracking."""

    id: str
    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    created_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_time: datetime | None = None
    completed_time: datetime | None = None
    status: ExperimentStatus = ExperimentStatus.CREATED
    parameters: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    git_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "created_time": self.created_time.isoformat(),
            "started_time": (self.started_time.isoformat() if self.started_time else None),
            "completed_time": (self.completed_time.isoformat() if self.completed_time else None),
            "status": self.status.value,
            "parameters": self.parameters,
            "environment": self.environment,
            "git_commit": self.git_commit,
        }


@dataclass
class ExperimentResult:
    """Container for experiment results."""

    experiment_id: str
    name: str
    value: Any
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "value": self._serialize_value(self.value),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def _serialize_value(self, value: Any) -> Any:
        """Serialize result value for storage."""
        if isinstance(value, np.ndarray):
            return {
                "type": "numpy_array",
                "shape": value.shape,
                "dtype": str(value.dtype),
                "data_file": f"result_{self.name}.npy",
            }
        elif hasattr(value, "to_dict"):
            return value.to_dict()
        elif isinstance(value, (dict, list, str, int, float, bool)):
            return value
        else:
            return str(value)


class Experiment:
    """
    Individual experiment with complete lifecycle management.

    An experiment represents a single computational run with specific
    parameters, tracking all inputs, outputs, and execution metadata.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        workspace_path: Path | None = None,
        tags: list[str] | None = None,
    ):
        """
        Initialize experiment.

        Args:
            name: Experiment name
            description: Experiment description
            workspace_path: Path to experiment workspace
            tags: List of tags for categorization
        """
        self.metadata = ExperimentMetadata(id=str(uuid.uuid4()), name=name, description=description, tags=tags or [])

        # Workspace setup
        self.workspace_path = workspace_path or Path.cwd() / ".mfg_experiments"
        self.experiment_dir = self.workspace_path / f"experiment_{self.metadata.id}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results: dict[str, ExperimentResult] = {}
        self.artifacts: dict[str, Path] = {}
        self.logs: list[dict[str, Any]] = []

        # Execution tracking
        self.execution_time: float | None = None
        self.error_message: str | None = None

        # Setup logging
        self.logger = self._setup_logging()

        # Capture environment
        self._capture_environment()

        self.logger.info(f"Created experiment '{name}' with ID {self.metadata.id}")

    def start(self):
        """Start experiment execution."""
        self.metadata.started_time = datetime.now(UTC)
        self.metadata.status = ExperimentStatus.RUNNING

        self.logger.info(f"Started experiment '{self.metadata.name}'")

    def complete(self):
        """Mark experiment as completed."""
        self.metadata.completed_time = datetime.now(UTC)
        self.metadata.status = ExperimentStatus.COMPLETED

        if self.metadata.started_time:
            self.execution_time = (self.metadata.completed_time - self.metadata.started_time).total_seconds()
            self.logger.info(f"Completed experiment '{self.metadata.name}' in {self.execution_time:.2f}s")
        else:
            self.logger.info(f"Completed experiment '{self.metadata.name}' (no start time recorded)")

    def fail(self, error_message: str):
        """Mark experiment as failed."""
        self.metadata.status = ExperimentStatus.FAILED
        self.error_message = error_message

        if self.metadata.started_time:
            failed_time = datetime.now(UTC)
            self.execution_time = (failed_time - self.metadata.started_time).total_seconds()

        self.logger.error(f"Experiment '{self.metadata.name}' failed: {error_message}")

    def cancel(self):
        """Cancel experiment execution."""
        self.metadata.status = ExperimentStatus.CANCELLED
        self.logger.warning(f"Cancelled experiment '{self.metadata.name}'")

    def set_parameter(self, name: str, value: Any):
        """Set experiment parameter."""
        self.metadata.parameters[name] = value
        self.logger.debug(f"Set parameter {name} = {value}")

    def set_parameters(self, parameters: dict[str, Any]):
        """Set multiple experiment parameters."""
        self.metadata.parameters.update(parameters)
        self.logger.debug(f"Set parameters: {parameters}")

    def add_result(self, name: str, value: Any, metadata: dict[str, Any] | None = None):
        """Add result to experiment."""
        result = ExperimentResult(
            experiment_id=self.metadata.id,
            name=name,
            value=value,
            metadata=metadata or {},
        )

        self.results[name] = result

        # Save large results separately
        if isinstance(value, np.ndarray):
            self._save_array_result(name, value)

        self.logger.debug(f"Added result '{name}'")

    def get_result(self, name: str) -> Any | None:
        """Get result by name."""
        if name in self.results:
            result = self.results[name]

            # Load array data if needed
            if isinstance(result.value, dict) and result.value.get("type") == "numpy_array":
                return self._load_array_result(name)

            return result.value

        return None

    def add_artifact(self, name: str, file_path: str | Path):
        """Add file artifact to experiment."""
        file_path = Path(file_path)

        if file_path.exists():
            # Copy to experiment directory
            artifact_path = self.experiment_dir / "artifacts" / file_path.name
            artifact_path.parent.mkdir(exist_ok=True)

            import shutil

            shutil.copy2(file_path, artifact_path)

            self.artifacts[name] = artifact_path
            self.logger.debug(f"Added artifact '{name}': {artifact_path}")
        else:
            self.logger.warning(f"Artifact file not found: {file_path}")

    def log_message(self, level: str, message: str, **kwargs):
        """Log message with experiment context."""
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level,
            "message": message,
            "data": kwargs,
        }

        self.logs.append(log_entry)

        # Also log to logger
        getattr(self.logger, level.lower())(message)

    def save(self):
        """Save experiment to disk."""
        # Save metadata
        metadata_file = self.experiment_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

        # Save results
        results_file = self.experiment_dir / "results.json"
        results_data = {name: result.to_dict() for name, result in self.results.items()}
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        # Save logs
        logs_file = self.experiment_dir / "logs.json"
        with open(logs_file, "w") as f:
            json.dump(self.logs, f, indent=2)

        # Save experiment state
        state_file = self.experiment_dir / "experiment.pkl"
        state_data = {
            "metadata": self.metadata,
            "artifacts": {name: str(path) for name, path in self.artifacts.items()},
            "execution_time": self.execution_time,
            "error_message": self.error_message,
        }

        with open(state_file, "wb") as f:
            pickle.dump(state_data, f)

        self.logger.info(f"Saved experiment to {self.experiment_dir}")

    def load(self, experiment_dir: Path):
        """Load experiment from disk."""
        self.experiment_dir = experiment_dir

        try:
            # Load metadata
            metadata_file = experiment_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata_dict = json.load(f)

                self.metadata = ExperimentMetadata(
                    id=metadata_dict["id"],
                    name=metadata_dict["name"],
                    description=metadata_dict["description"],
                    tags=metadata_dict.get("tags", []),
                    created_time=datetime.fromisoformat(metadata_dict["created_time"]),
                    started_time=(
                        datetime.fromisoformat(metadata_dict["started_time"])
                        if metadata_dict.get("started_time")
                        else None
                    ),
                    completed_time=(
                        datetime.fromisoformat(metadata_dict["completed_time"])
                        if metadata_dict.get("completed_time")
                        else None
                    ),
                    status=ExperimentStatus(metadata_dict["status"]),
                    parameters=metadata_dict.get("parameters", {}),
                    environment=metadata_dict.get("environment", {}),
                    git_commit=metadata_dict.get("git_commit"),
                )

            # Load results
            results_file = experiment_dir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    results_data = json.load(f)

                for name, result_dict in results_data.items():
                    result = ExperimentResult(
                        experiment_id=result_dict["experiment_id"],
                        name=result_dict["name"],
                        value=result_dict["value"],
                        timestamp=datetime.fromisoformat(result_dict["timestamp"]),
                        metadata=result_dict.get("metadata", {}),
                    )
                    self.results[name] = result

            # Load logs
            logs_file = experiment_dir / "logs.json"
            if logs_file.exists():
                with open(logs_file) as f:
                    self.logs = json.load(f)

            # Load experiment state
            state_file = experiment_dir / "experiment.pkl"
            if state_file.exists():
                with open(state_file, "rb") as f:
                    state_data = pickle.load(f)

                self.execution_time = state_data.get("execution_time")
                self.error_message = state_data.get("error_message")

                # Restore artifact paths
                for name, path_str in state_data.get("artifacts", {}).items():
                    self.artifacts[name] = Path(path_str)

            self.logger.info(f"Loaded experiment '{self.metadata.name}' from {experiment_dir}")

        except Exception as e:
            self.logger.error(f"Failed to load experiment: {e}")
            raise

    def compare_with(self, other: Experiment) -> dict[str, Any]:
        """Compare this experiment with another."""
        comparison = {
            "experiments": {
                "this": {
                    "id": self.metadata.id,
                    "name": self.metadata.name,
                    "status": self.metadata.status.value,
                    "execution_time": self.execution_time,
                },
                "other": {
                    "id": other.metadata.id,
                    "name": other.metadata.name,
                    "status": other.metadata.status.value,
                    "execution_time": other.execution_time,
                },
            },
            "parameter_differences": {},
            "result_differences": {},
            "performance_comparison": {},
        }

        # Compare parameters
        all_param_names = set(self.metadata.parameters.keys()) | set(other.metadata.parameters.keys())
        for param_name in all_param_names:
            this_value = self.metadata.parameters.get(param_name, "NOT_SET")
            other_value = other.metadata.parameters.get(param_name, "NOT_SET")

            if this_value != other_value:
                comparison["parameter_differences"][param_name] = {
                    "this": this_value,
                    "other": other_value,
                }

        # Compare results
        all_result_names = set(self.results.keys()) | set(other.results.keys())
        for result_name in all_result_names:
            this_result = self.get_result(result_name)
            other_result = other.get_result(result_name)

            if this_result is not None and other_result is not None:
                # Numeric comparison
                if isinstance(this_result, (int, float)) and isinstance(other_result, (int, float)):
                    comparison["result_differences"][result_name] = {
                        "this": this_result,
                        "other": other_result,
                        "absolute_difference": abs(this_result - other_result),
                        "relative_difference": abs(this_result - other_result)
                        / max(abs(this_result), abs(other_result), 1e-10),
                    }
                # Array comparison
                elif (
                    isinstance(this_result, np.ndarray)
                    and isinstance(other_result, np.ndarray)
                    and this_result.shape == other_result.shape
                ):
                    comparison["result_differences"][result_name] = {
                        "shape": this_result.shape,
                        "max_absolute_difference": np.max(np.abs(this_result - other_result)),
                        "mean_absolute_difference": np.mean(np.abs(this_result - other_result)),
                        "relative_error": np.linalg.norm(this_result - other_result)
                        / max(
                            np.linalg.norm(this_result),
                            np.linalg.norm(other_result),
                            1e-10,
                        ),
                    }
            elif this_result is None and other_result is not None:
                comparison["result_differences"][result_name] = {"missing_in": "this"}
            elif this_result is not None and other_result is None:
                comparison["result_differences"][result_name] = {"missing_in": "other"}

        # Performance comparison
        if self.execution_time is not None and other.execution_time is not None:
            comparison["performance_comparison"] = {
                "execution_time_ratio": self.execution_time / other.execution_time,
                "absolute_time_difference": abs(self.execution_time - other.execution_time),
            }

        return comparison

    def _save_array_result(self, name: str, array: np.ndarray):
        """Save numpy array result to separate file."""
        array_dir = self.experiment_dir / "arrays"
        array_dir.mkdir(exist_ok=True)

        array_file = array_dir / f"result_{name}.npy"
        np.save(array_file, array)

    def _load_array_result(self, name: str) -> np.ndarray | None:
        """Load numpy array result from file."""
        array_file = self.experiment_dir / "arrays" / f"result_{name}.npy"

        if array_file.exists():
            return np.load(array_file)

        return None

    def _capture_environment(self):
        """Capture current environment information."""
        import os
        import platform
        import sys

        self.metadata.environment = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "working_directory": str(Path.cwd()),
            "environment_variables": {
                key: value for key, value in os.environ.items() if key.startswith(("MFG_", "PYTHON", "PATH"))
            },
        }

        # Try to get git commit
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )
            if result.returncode == 0:
                self.metadata.git_commit = result.stdout.strip()
        except Exception:
            pass

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the experiment."""
        logger = logging.getLogger(f"mfg_experiment.{self.metadata.id}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # File handler
            log_file = self.experiment_dir / "experiment.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)

            # Formatter
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)

            logger.addHandler(file_handler)

        return logger


class ExperimentTracker:
    """
    Central tracker for managing multiple experiments.

    Provides experiment discovery, organization, and comparative analysis
    across research sessions and parameter studies.
    """

    def __init__(self, workspace_path: Path | None = None):
        """
        Initialize experiment tracker.

        Args:
            workspace_path: Path to experiments workspace
        """
        self.workspace_path = workspace_path or Path.cwd() / ".mfg_experiments"
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        self.experiments: dict[str, Experiment] = {}
        self.logger = self._setup_logging()

        # Load existing experiments
        self._load_existing_experiments()

        self.logger.info(f"Initialized experiment tracker with workspace: {self.workspace_path}")

    def create_experiment(
        self,
        name: str,
        description: str = "",
        tags: list[str] | None = None,
        workflow: Workflow | None = None,
    ) -> Experiment:
        """Create a new experiment."""
        experiment = Experiment(name, description, self.workspace_path, tags)

        # Link to workflow if provided
        if workflow is not None:
            experiment.set_parameter("workflow_id", workflow.id)
            experiment.set_parameter("workflow_name", workflow.name)

        self.experiments[experiment.metadata.id] = experiment

        self.logger.info(f"Created experiment '{name}' with ID {experiment.metadata.id}")
        return experiment

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Get experiment by ID."""
        return self.experiments.get(experiment_id)

    def list_experiments(
        self,
        tags: list[str] | None = None,
        status: ExperimentStatus | None = None,
    ) -> list[dict[str, Any]]:
        """List experiments with optional filtering."""
        experiments = []

        for experiment in self.experiments.values():
            # Apply filters
            if tags and not any(tag in experiment.metadata.tags for tag in tags):
                continue

            if status and experiment.metadata.status != status:
                continue

            experiments.append(
                {
                    "id": experiment.metadata.id,
                    "name": experiment.metadata.name,
                    "description": experiment.metadata.description,
                    "tags": experiment.metadata.tags,
                    "status": experiment.metadata.status.value,
                    "created_time": experiment.metadata.created_time,
                    "execution_time": experiment.execution_time,
                    "num_results": len(experiment.results),
                }
            )

        # Sort by creation time (newest first)
        experiments.sort(key=lambda x: x["created_time"], reverse=True)

        return experiments

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete experiment and its data."""
        if experiment_id not in self.experiments:
            return False

        experiment = self.experiments[experiment_id]

        # Remove from memory
        del self.experiments[experiment_id]

        # Remove from disk
        import shutil

        if experiment.experiment_dir.exists():
            shutil.rmtree(experiment.experiment_dir)

        self.logger.info(f"Deleted experiment {experiment_id}")
        return True

    def compare_experiments(self, experiment_ids: list[str]) -> dict[str, Any]:
        """Compare multiple experiments."""
        experiments = [self.get_experiment(exp_id) for exp_id in experiment_ids]
        experiments = [exp for exp in experiments if exp is not None]

        if len(experiments) < 2:
            return {"error": "Need at least 2 experiments for comparison"}

        comparison = {
            "experiment_count": len(experiments),
            "experiments": [
                {
                    "id": exp.metadata.id,
                    "name": exp.metadata.name,
                    "status": exp.metadata.status.value,
                    "execution_time": exp.execution_time,
                }
                for exp in experiments
            ],
            "pairwise_comparisons": {},
            "aggregate_analysis": {},
        }

        # Pairwise comparisons
        for i, exp1 in enumerate(experiments):
            for _, exp2 in enumerate(experiments[i + 1 :], i + 1):
                key = f"{exp1.metadata.name}_vs_{exp2.metadata.name}"
                comparison["pairwise_comparisons"][key] = exp1.compare_with(exp2)

        # Aggregate analysis
        comparison["aggregate_analysis"] = self._analyze_experiment_group(experiments)

        return comparison

    def find_similar_experiments(self, experiment_id: str, similarity_threshold: float = 0.8) -> list[dict[str, Any]]:
        """Find experiments similar to the given one."""
        target_experiment = self.get_experiment(experiment_id)
        if target_experiment is None:
            return []

        similar_experiments = []

        for exp_id, experiment in self.experiments.items():
            if exp_id == experiment_id:
                continue

            similarity = self._compute_experiment_similarity(target_experiment, experiment)

            if similarity >= similarity_threshold:
                similar_experiments.append(
                    {
                        "experiment": {
                            "id": experiment.metadata.id,
                            "name": experiment.metadata.name,
                            "status": experiment.metadata.status.value,
                        },
                        "similarity_score": similarity,
                    }
                )

        # Sort by similarity (highest first)
        similar_experiments.sort(key=lambda x: x["similarity_score"], reverse=True)

        return similar_experiments

    def export_experiments(self, experiment_ids: list[str], export_format: str = "json") -> str:
        """Export experiments to specified format."""
        experiments_data = []

        for exp_id in experiment_ids:
            experiment = self.get_experiment(exp_id)
            if experiment is not None:
                exp_data = {
                    "metadata": experiment.metadata.to_dict(),
                    "results": {name: result.to_dict() for name, result in experiment.results.items()},
                    "execution_time": experiment.execution_time,
                    "error_message": experiment.error_message,
                }
                experiments_data.append(exp_data)

        # Export based on format
        timestamp = int(time.time())

        if export_format == "json":
            export_file = self.workspace_path / f"export_{timestamp}.json"
            with open(export_file, "w") as f:
                json.dump(experiments_data, f, indent=2)

        elif export_format == "csv":
            # Flatten data for CSV export
            import pandas as pd

            flattened_data = []
            for exp_data in experiments_data:
                flat_row = {}
                flat_row.update(exp_data["metadata"])

                # Add result values
                for result_name, result_data in exp_data["results"].items():
                    if isinstance(result_data["value"], (int, float, str)):
                        flat_row[f"result_{result_name}"] = result_data["value"]

                flat_row["execution_time"] = exp_data["execution_time"]
                flattened_data.append(flat_row)

            df = pd.DataFrame(flattened_data)
            export_file = self.workspace_path / f"export_{timestamp}.csv"
            df.to_csv(export_file, index=False)

        else:
            raise ValueError(f"Unsupported export format: {export_format}")

        self.logger.info(f"Exported {len(experiments_data)} experiments to {export_file}")
        return str(export_file)

    def _load_existing_experiments(self):
        """Load existing experiments from workspace."""
        if not self.workspace_path.exists():
            return

        for experiment_dir in self.workspace_path.glob("experiment_*"):
            if experiment_dir.is_dir():
                try:
                    experiment = Experiment("temp", workspace_path=self.workspace_path)
                    experiment.load(experiment_dir)
                    self.experiments[experiment.metadata.id] = experiment

                except Exception as e:
                    self.logger.warning(f"Could not load experiment from {experiment_dir}: {e}")

    def _analyze_experiment_group(self, experiments: list[Experiment]) -> dict[str, Any]:
        """Analyze a group of experiments for aggregate patterns."""
        analysis = {
            "execution_times": [],
            "success_rate": 0,
            "common_parameters": {},
            "varying_parameters": {},
            "result_statistics": {},
        }

        # Collect execution times
        execution_times = [exp.execution_time for exp in experiments if exp.execution_time is not None]
        if execution_times:
            analysis["execution_times"] = {
                "mean": np.mean(execution_times),
                "std": np.std(execution_times),
                "min": np.min(execution_times),
                "max": np.max(execution_times),
            }

        # Success rate
        successful = sum(1 for exp in experiments if exp.metadata.status == ExperimentStatus.COMPLETED)
        analysis["success_rate"] = successful / len(experiments)

        # Parameter analysis
        all_params = {}
        for experiment in experiments:
            for param_name, param_value in experiment.metadata.parameters.items():
                if param_name not in all_params:
                    all_params[param_name] = []
                all_params[param_name].append(param_value)

        for param_name, values in all_params.items():
            unique_values = set(values)
            if len(unique_values) == 1:
                analysis["common_parameters"][param_name] = next(iter(unique_values))
            else:
                analysis["varying_parameters"][param_name] = {
                    "unique_values": list(unique_values),
                    "distribution": {val: values.count(val) for val in unique_values},
                }

        return analysis

    def _compute_experiment_similarity(self, exp1: Experiment, exp2: Experiment) -> float:
        """Compute similarity score between two experiments."""
        # Simple similarity based on parameter overlap
        params1 = set(exp1.metadata.parameters.items())
        params2 = set(exp2.metadata.parameters.items())

        if not params1 and not params2:
            return 1.0

        intersection = params1 & params2
        union = params1 | params2

        jaccard_similarity = len(intersection) / len(union) if union else 0.0

        # Weight by tag overlap
        tags1 = set(exp1.metadata.tags)
        tags2 = set(exp2.metadata.tags)

        tag_intersection = tags1 & tags2
        tag_union = tags1 | tags2

        tag_similarity = len(tag_intersection) / len(tag_union) if tag_union else 0.0

        # Combined similarity
        return 0.7 * jaccard_similarity + 0.3 * tag_similarity

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the experiment tracker."""
        logger = logging.getLogger("mfg_experiment_tracker")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # File handler
            log_file = self.workspace_path / "experiment_tracker.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)

            # Formatter
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)

            logger.addHandler(file_handler)

        return logger


# Export main classes
__all__ = [
    "Experiment",
    "ExperimentMetadata",
    "ExperimentResult",
    "ExperimentStatus",
    "ExperimentTracker",
]
