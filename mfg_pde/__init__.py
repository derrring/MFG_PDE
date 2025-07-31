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
from .core.mfg_problem import (
    create_mfg_problem,
    ExampleMFGProblem,
    MFGComponents,
    MFGProblem,
    MFGProblemBuilder,
)
from .core.network_mfg_problem import (
    create_grid_mfg_problem,
    create_random_mfg_problem,
    create_scale_free_mfg_problem,
    NetworkMFGComponents,
    NetworkMFGProblem,
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
    create_general_mfg_problem,
    GeneralMFGFactory,
    get_general_factory,
)
from .geometry import BoundaryConditions
from .geometry.network_backend import (
    get_backend_manager,
    NetworkBackendType,
    OperationType,
    set_preferred_backend,
)
from .geometry.network_geometry import (
    compute_network_statistics,
    create_network,
    GridNetwork,
    NetworkData,
    NetworkType,
    RandomNetwork,
    ScaleFreeNetwork,
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
    from .utils.notebook_reporting import (
        create_comparative_analysis,
        create_mfg_research_report,
        MFGNotebookReporter,
    )

    NOTEBOOK_REPORTING_AVAILABLE = True
except ImportError:
    NOTEBOOK_REPORTING_AVAILABLE = False
