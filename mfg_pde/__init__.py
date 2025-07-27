from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mfg_pde")  # Matches the name in pyproject.toml
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0-dev"  # Or some other placeholde

from .config import (
    create_accurate_config,
    create_fast_config,
    create_research_config,
    MFGSolverConfig,
)
from .core.boundaries import BoundaryConditions
from .core.mfg_problem import ExampleMFGProblem, MFGProblem
from .factory import (
    create_accurate_solver,
    create_fast_solver,
    create_monitored_solver,
    create_research_solver,
    create_solver,
    SolverFactory,
)

# Interactive research reporting (optional dependency)
try:
    from .utils.notebook_reporting import (
        create_comparative_analysis,
        create_mfg_research_report,
        MFGNotebookReporter,
    )

    NOTEBOOK_REPORTING_AVAILABLE = True
except ImportError:
    NOTEBOOK_REPORTING_AVAILABLE = False
