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
from .geometry import BoundaryConditions
from .geometry.network_geometry import (
    NetworkData, NetworkType, GridNetwork, RandomNetwork, ScaleFreeNetwork,
    create_network, compute_network_statistics
)
from .core.mfg_problem import MFGProblem, ExampleMFGProblem, MFGProblemBuilder, MFGComponents, create_mfg_problem
from .core.network_mfg_problem import (
    NetworkMFGProblem, NetworkMFGComponents,
    create_grid_mfg_problem, create_random_mfg_problem, create_scale_free_mfg_problem
)
from .factory import (
    create_accurate_solver,
    create_fast_solver,
    create_monitored_solver,
    create_research_solver,
    create_solver,
    SolverFactory,
)
from .factory.general_mfg_factory import (
    GeneralMFGFactory,
    get_general_factory,
    create_general_mfg_problem,
)

# Geometry system for 2D/3D complex domains (optional dependency)
try:
    from .geometry import (
        BaseGeometry,
        MeshData,
        Domain2D,
        MeshPipeline,
        MeshManager,
        BoundaryManager,
        GeometricBoundaryCondition,
    )
    
    GEOMETRY_SYSTEM_AVAILABLE = True
except ImportError:
    GEOMETRY_SYSTEM_AVAILABLE = False

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
