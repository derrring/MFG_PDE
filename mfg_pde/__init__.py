from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mfg_pde")  # Matches the name in pyproject.toml
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0-dev"  # Or some other placeholder

# Ensure NumPy 2.0+ compatibility on import
from .utils.numpy_compat import ensure_numpy_compatibility

_numpy_info = ensure_numpy_compatibility()

from .alg.numerical.fp_solvers.fp_particle import KDENormalization  # noqa: E402
from .config import MFGSolverConfig  # noqa: E402
from .core.mfg_problem import (  # noqa: E402
    MFGComponents,
    MFGProblem,
)
from .extensions.topology import (  # noqa: E402
    NetworkMFGComponents,
    NetworkMFGProblem,
    create_grid_mfg_problem,
    create_random_mfg_problem,
    create_scale_free_mfg_problem,
)
from .factory import (  # noqa: E402
    SolverFactory,
    create_solver,
)
from .factory.general_mfg_factory import (  # noqa: E402
    GeneralMFGFactory,
    create_general_mfg_problem,
    get_general_factory,
)
from .geometry import (  # noqa: E402
    BoundaryConditions,
    GridNetwork,
    NetworkBackendType,
    NetworkData,
    NetworkType,
    OperationType,
    RandomNetwork,
    ScaleFreeNetwork,
    compute_network_statistics,
    create_network,
    get_backend_manager,
    set_preferred_backend,
)
from .solvers.variational import (  # noqa: E402
    VariationalMFGComponents,
    VariationalMFGProblem,
    create_obstacle_variational_mfg,
    create_quadratic_variational_mfg,
)

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

# High-dimensional MFG capabilities - REMOVED in v0.14.0
# Use MFGProblem directly with spatial_bounds and spatial_discretization parameters
# See: docs/migration/DEPRECATION_MODERNIZATION_GUIDE.md
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


# Optional dependency utilities
def show_optional_features() -> None:
    """
    Display status of all optional features and dependencies.

    Prints a formatted table showing which optional dependencies are installed
    and available, along with installation instructions for missing features.

    Example:
        >>> import mfg_pde
        >>> mfg_pde.show_optional_features()
        MFG_PDE Optional Features
        ==================================================
        pytorch        : ✓ Available
        jax            : ✗ Not installed
        ...
    """
    from .utils.dependencies import show_optional_features as _show_features

    _show_features()


# Public API exports
__all__ = [
    # Geometry
    "BoundaryConditions",
    # General MFG factory
    "GeneralMFGFactory",
    # Network geometry
    "GridNetwork",
    # Solver enums and configuration
    "KDENormalization",
    "MFGComponents",
    "MFGProblem",
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
    "create_general_mfg_problem",
    "create_grid_mfg_problem",
    "create_network",
    "create_obstacle_variational_mfg",
    "create_quadratic_variational_mfg",
    "create_random_mfg_problem",
    "create_scale_free_mfg_problem",
    "create_solver",
    "get_backend_manager",
    "get_general_factory",
    "set_preferred_backend",
    "show_optional_features",
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

# GridBasedMFGProblem, HighDimMFGProblem, HybridMFGSolver removed in v0.14.0
# Use MFGProblem directly instead

if NOTEBOOK_REPORTING_AVAILABLE:
    __all__.extend(
        [
            "MFGNotebookReporter",
            "create_comparative_analysis",
            "create_mfg_research_report",
        ]
    )
