from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("mfg_pde") # Matches the name in pyproject.toml
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0-dev" # Or some other placeholde

from .core.mfg_problem import MFGProblem
from .core.base_solver import MFGSolver
from .core.boundaries import BoundaryConditions