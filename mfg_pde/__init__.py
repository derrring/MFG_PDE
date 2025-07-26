from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("mfg_pde") # Matches the name in pyproject.toml
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0-dev" # Or some other placeholde

from .core.mfg_problem import MFGProblem, ExampleMFGProblem
from .core.boundaries import BoundaryConditions
from .config import MFGSolverConfig, create_default_config, create_fast_config, create_accurate_config, create_research_config
from .factory import (
    SolverFactory, create_solver, create_fast_solver, 
    create_accurate_solver, create_research_solver, create_monitored_solver
)