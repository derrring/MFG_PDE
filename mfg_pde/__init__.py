from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mfg_pde")  # Matches the name in pyproject.toml
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0-dev"  # Or some other placeholder

# Ensure NumPy 2.0+ compatibility on import
from .utils.numpy_compat import ensure_numpy_compatibility

_numpy_info = ensure_numpy_compatibility()

from .config import MFGSolverConfig, create_accurate_config, create_fast_config, create_research_config
from .core.lagrangian_mfg_problem import (
    LagrangianComponents,
    LagrangianMFGProblem,
    create_obstacle_lagrangian_mfg,
    create_quadratic_lagrangian_mfg,
)
from .core.mfg_problem import ExampleMFGProblem, MFGComponents, MFGProblem, MFGProblemBuilder, create_mfg_problem
from .core.network_mfg_problem import (
    NetworkMFGComponents,
    NetworkMFGProblem,
    create_grid_mfg_problem,
    create_random_mfg_problem,
    create_scale_free_mfg_problem,
)
from .factory import (
    SolverFactory,
    create_accurate_solver,
    create_fast_solver,
    create_monitored_solver,
    create_research_solver,
    create_solver,
)
from .factory.general_mfg_factory import GeneralMFGFactory, create_general_mfg_problem, get_general_factory
from .geometry import BoundaryConditions
from .geometry.network_backend import NetworkBackendType, OperationType, get_backend_manager, set_preferred_backend
from .geometry.network_geometry import (
    GridNetwork,
    NetworkData,
    NetworkType,
    RandomNetwork,
    ScaleFreeNetwork,
    compute_network_statistics,
    create_network,
)

# Geometry system for 2D/3D complex domains (optional dependency)
try:
    from .geometry import (
        BaseGeometry,
        BoundaryManager,
        Domain2D,
        GeometricBoundaryCondition,
        MeshData,
        MeshManager,
        MeshPipeline,
    )

    GEOMETRY_SYSTEM_AVAILABLE = True
except ImportError:
    GEOMETRY_SYSTEM_AVAILABLE = False

# Interactive research reporting (optional dependency)
try:
    from .utils.notebook_reporting import MFGNotebookReporter, create_comparative_analysis, create_mfg_research_report

    NOTEBOOK_REPORTING_AVAILABLE = True
except ImportError:
    NOTEBOOK_REPORTING_AVAILABLE = False
