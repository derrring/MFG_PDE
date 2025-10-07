from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mfg_pde")  # Matches the name in pyproject.toml
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0-dev"  # Or some other placeholder

# Ensure NumPy 2.0+ compatibility on import
from .utils.numpy_compat import ensure_numpy_compatibility

_numpy_info = ensure_numpy_compatibility()

from .config import MFGSolverConfig, create_accurate_config, create_fast_config, create_research_config  # noqa: E402
from .core.mfg_problem import (  # noqa: E402
    ExampleMFGProblem,
    MFGComponents,
    MFGProblem,
    MFGProblemBuilder,
    create_mfg_problem,
)
from .core.network_mfg_problem import (  # noqa: E402
    NetworkMFGComponents,
    NetworkMFGProblem,
    create_grid_mfg_problem,
    create_random_mfg_problem,
    create_scale_free_mfg_problem,
)
from .core.variational_mfg_problem import (  # noqa: E402
    VariationalMFGComponents,
    VariationalMFGProblem,
    create_obstacle_variational_mfg,
    create_quadratic_variational_mfg,
)
from .factory import (  # noqa: E402
    SolverFactory,
    create_accurate_solver,
    create_fast_solver,
    create_monitored_solver,
    create_research_solver,
    create_solver,
    create_standard_solver,
)
from .factory.general_mfg_factory import (  # noqa: E402
    GeneralMFGFactory,
    create_general_mfg_problem,
    get_general_factory,
)
from .geometry import BoundaryConditions  # noqa: E402
from .geometry.network_backend import (  # noqa: E402
    NetworkBackendType,
    OperationType,
    get_backend_manager,
    set_preferred_backend,
)
from .geometry.network_geometry import (  # noqa: E402
    GridNetwork,
    NetworkData,
    NetworkType,
    RandomNetwork,
    ScaleFreeNetwork,
    compute_network_statistics,
    create_network,
)

# Simple API removed - use factory API instead
# from .simple import ...

# Geometry system for 2D/3D complex domains (optional dependency)
try:
    from .geometry import (
        BaseGeometry,  # noqa: F401
        BoundaryManager,  # noqa: F401
        Domain2D,  # noqa: F401
        Domain3D,  # noqa: F401
        GeometricBoundaryCondition,  # noqa: F401
        MeshData,  # noqa: F401
        MeshManager,  # noqa: F401
        MeshPipeline,  # noqa: F401
    )

    GEOMETRY_SYSTEM_AVAILABLE = True
except ImportError:
    GEOMETRY_SYSTEM_AVAILABLE = False

# High-dimensional MFG capabilities
try:
    from .core.highdim_mfg_problem import (
        GridBasedMFGProblem,  # noqa: F401
        HighDimMFGProblem,  # noqa: F401
        HybridMFGSolver,  # noqa: F401
    )

    HIGHDIM_MFG_AVAILABLE = True
except ImportError:
    HIGHDIM_MFG_AVAILABLE = False

# Interactive research reporting (optional dependency)
try:
    from .utils.notebooks.reporting import (  # noqa: F401
        MFGNotebookReporter,
        create_comparative_analysis,
        create_mfg_research_report,
    )

    NOTEBOOK_REPORTING_AVAILABLE = True
except ImportError:
    NOTEBOOK_REPORTING_AVAILABLE = False


# Public API exports
__all__ = [
    # Geometry
    "BoundaryConditions",
    # Core MFG classes
    "ExampleMFGProblem",
    # General MFG factory
    "GeneralMFGFactory",
    # Network geometry
    "GridNetwork",
    "MFGComponents",
    "MFGProblem",
    "MFGProblemBuilder",
    # Configuration
    "MFGSolverConfig",
    # Network backend
    "NetworkBackendType",
    "NetworkData",
    # Network MFG
    "NetworkMFGComponents",
    "NetworkMFGProblem",
    "NetworkType",
    "OperationType",
    "RandomNetwork",
    "ScaleFreeNetwork",
    # Factory methods
    "SolverFactory",
    # Variational MFG
    "VariationalMFGComponents",
    "VariationalMFGProblem",
    "compute_network_statistics",
    "create_accurate_config",
    "create_accurate_solver",
    "create_fast_config",
    "create_fast_solver",
    "create_general_mfg_problem",
    "create_grid_mfg_problem",
    "create_mfg_problem",
    "create_monitored_solver",
    "create_network",
    "create_obstacle_variational_mfg",
    "create_quadratic_variational_mfg",
    "create_random_mfg_problem",
    "create_research_config",
    "create_research_solver",
    "create_scale_free_mfg_problem",
    "create_solver",
    "create_standard_solver",
    "get_backend_manager",
    "get_general_factory",
    "set_preferred_backend",
]

# Add conditionally available imports to __all__ when available
if GEOMETRY_SYSTEM_AVAILABLE:
    __all__.extend(
        [
            "BaseGeometry",
            "BoundaryManager",
            "Domain2D",
            "Domain3D",
            "GeometricBoundaryCondition",
            "MeshData",
            "MeshManager",
            "MeshPipeline",
        ]
    )

if HIGHDIM_MFG_AVAILABLE:
    __all__.extend(
        [
            "GridBasedMFGProblem",
            "HighDimMFGProblem",
            "HybridMFGSolver",
        ]
    )

if NOTEBOOK_REPORTING_AVAILABLE:
    __all__.extend(
        [
            "MFGNotebookReporter",
            "create_comparative_analysis",
            "create_mfg_research_report",
        ]
    )
