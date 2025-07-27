"""
Structured configuration classes for MFG_PDE solvers.

This module provides dataclass-based configuration objects that replace scattered
constructor parameters with organized, validated, and well-documented settings.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any, Literal
import warnings


@dataclass
class NewtonConfig:
    """
    Configuration for Newton-type iterative methods used in HJB solvers.

    Attributes:
        max_iterations: Maximum number of Newton iterations
        tolerance: Convergence tolerance for Newton method
        damping_factor: Damping parameter for Newton updates (0 < damping <= 1)
        line_search: Whether to use line search for step size control
        verbose: Whether to print Newton iteration details
    """

    max_iterations: int = 30
    tolerance: float = 1e-6
    damping_factor: float = 1.0
    line_search: bool = False
    verbose: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.tolerance <= 0:
            raise ValueError(f"tolerance must be > 0, got {self.tolerance}")
        if not (0 < self.damping_factor <= 1):
            raise ValueError(
                f"damping_factor must be in (0, 1], got {self.damping_factor}"
            )

    @classmethod
    def fast(cls) -> "NewtonConfig":
        """Create configuration optimized for speed."""
        return cls(max_iterations=10, tolerance=1e-4, damping_factor=0.8)

    @classmethod
    def accurate(cls) -> "NewtonConfig":
        """Create configuration optimized for accuracy."""
        return cls(max_iterations=50, tolerance=1e-8, damping_factor=1.0)


@dataclass
class PicardConfig:
    """
    Configuration for Picard (fixed-point) iterations in MFG solvers.

    Attributes:
        max_iterations: Maximum number of Picard iterations
        tolerance: Convergence tolerance for Picard iteration
        damping_factor: Damping parameter for Picard updates (0 < damping <= 1)
        convergence_check_frequency: How often to check convergence (every N iterations)
        verbose: Whether to print Picard iteration details
    """

    max_iterations: int = 20
    tolerance: float = 1e-5
    damping_factor: float = 0.5
    convergence_check_frequency: int = 1
    verbose: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.tolerance <= 0:
            raise ValueError(f"tolerance must be > 0, got {self.tolerance}")
        if not (0 < self.damping_factor <= 1):
            raise ValueError(
                f"damping_factor must be in (0, 1], got {self.damping_factor}"
            )
        if self.convergence_check_frequency < 1:
            raise ValueError(
                f"convergence_check_frequency must be >= 1, got {self.convergence_check_frequency}"
            )

    @classmethod
    def fast(cls) -> "PicardConfig":
        """Create configuration optimized for speed."""
        return cls(max_iterations=10, tolerance=1e-3, damping_factor=0.7)

    @classmethod
    def accurate(cls) -> "PicardConfig":
        """Create configuration optimized for accuracy."""
        return cls(max_iterations=50, tolerance=1e-7, damping_factor=0.3)


@dataclass
class GFDMConfig:
    """
    Configuration for Generalized Finite Difference Method (GFDM) solvers.

    Attributes:
        delta: Neighborhood radius for GFDM stencil
        taylor_order: Order of Taylor expansion for GFDM approximation
        weight_function: Weight function type for GFDM
        weight_scale: Scale parameter for weight function
        use_qp_constraints: Whether to use quadratic programming constraints
        boundary_method: Method for handling boundary conditions
    """

    delta: float = 0.1
    taylor_order: int = 2
    weight_function: Literal["gaussian", "inverse_distance", "uniform", "wendland"] = (
        "gaussian"
    )
    weight_scale: float = 1.0
    use_qp_constraints: bool = False
    boundary_method: Literal["dirichlet", "neumann", "extrapolation"] = "dirichlet"

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.delta <= 0:
            raise ValueError(f"delta must be > 0, got {self.delta}")
        if self.taylor_order < 1:
            raise ValueError(f"taylor_order must be >= 1, got {self.taylor_order}")
        if self.weight_scale <= 0:
            raise ValueError(f"weight_scale must be > 0, got {self.weight_scale}")

    @classmethod
    def fast(cls) -> "GFDMConfig":
        """Create configuration optimized for speed."""
        return cls(delta=0.15, taylor_order=1, weight_function="uniform")

    @classmethod
    def accurate(cls) -> "GFDMConfig":
        """Create configuration optimized for accuracy."""
        return cls(
            delta=0.05,
            taylor_order=3,
            weight_function="wendland",
            use_qp_constraints=True,
        )


@dataclass
class ParticleConfig:
    """
    Configuration for particle-based Fokker-Planck solvers.

    Attributes:
        num_particles: Number of particles for the simulation
        kde_bandwidth: Bandwidth method for kernel density estimation
        normalize_output: Whether to normalize KDE output
        boundary_handling: Method for handling particle boundary conditions
        random_seed: Random seed for reproducible particle initialization
    """

    num_particles: int = 5000
    kde_bandwidth: Union[str, float] = "scott"
    normalize_output: bool = True
    boundary_handling: Literal["absorbing", "reflecting", "periodic"] = "absorbing"
    random_seed: Optional[int] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_particles < 10:
            raise ValueError(f"num_particles must be >= 10, got {self.num_particles}")
        if isinstance(self.kde_bandwidth, (int, float)) and self.kde_bandwidth <= 0:
            raise ValueError(
                f"kde_bandwidth must be > 0 when numeric, got {self.kde_bandwidth}"
            )

    @classmethod
    def fast(cls) -> "ParticleConfig":
        """Create configuration optimized for speed."""
        return cls(num_particles=1000, kde_bandwidth="scott", normalize_output=False)

    @classmethod
    def accurate(cls) -> "ParticleConfig":
        """Create configuration optimized for accuracy."""
        return cls(num_particles=10000, kde_bandwidth=0.01, normalize_output=True)


@dataclass
class HJBConfig:
    """
    Configuration for Hamilton-Jacobi-Bellman equation solvers.

    Attributes:
        newton: Newton method configuration
        gfdm: GFDM method configuration (when applicable)
        solver_type: Type of HJB solver to use
        boundary_conditions: Boundary condition specifications
    """

    newton: NewtonConfig = field(default_factory=NewtonConfig)
    gfdm: GFDMConfig = field(default_factory=GFDMConfig)
    solver_type: Literal["fdm", "gfdm", "gfdm_qp", "semi_lagrangian"] = "gfdm"
    boundary_conditions: Optional[Dict[str, Any]] = None

    @classmethod
    def fast(cls) -> "HJBConfig":
        """Create configuration optimized for speed."""
        return cls(
            newton=NewtonConfig.fast(), gfdm=GFDMConfig.fast(), solver_type="fdm"
        )

    @classmethod
    def accurate(cls) -> "HJBConfig":
        """Create configuration optimized for accuracy."""
        return cls(
            newton=NewtonConfig.accurate(),
            gfdm=GFDMConfig.accurate(),
            solver_type="gfdm_qp",
        )


@dataclass
class FPConfig:
    """
    Configuration for Fokker-Planck equation solvers.

    Attributes:
        particle: Particle method configuration (when applicable)
        solver_type: Type of FP solver to use
        boundary_conditions: Boundary condition specifications
    """

    particle: ParticleConfig = field(default_factory=ParticleConfig)
    solver_type: Literal["fdm", "particle"] = "fdm"
    boundary_conditions: Optional[Dict[str, Any]] = None

    @classmethod
    def fast(cls) -> "FPConfig":
        """Create configuration optimized for speed."""
        return cls(particle=ParticleConfig.fast(), solver_type="fdm")

    @classmethod
    def accurate(cls) -> "FPConfig":
        """Create configuration optimized for accuracy."""
        return cls(particle=ParticleConfig.accurate(), solver_type="particle")


@dataclass
class MFGSolverConfig:
    """
    Complete configuration for MFG solver systems.

    This is the main configuration class that contains all settings for
    solving Mean Field Game systems.

    Attributes:
        picard: Picard iteration configuration
        hjb: Hamilton-Jacobi-Bellman solver configuration
        fp: Fokker-Planck solver configuration
        warm_start: Whether to enable warm start capability
        return_structured: Whether to return structured result objects
        metadata: Additional solver-specific metadata
    """

    picard: PicardConfig = field(default_factory=PicardConfig)
    hjb: HJBConfig = field(default_factory=HJBConfig)
    fp: FPConfig = field(default_factory=FPConfig)
    warm_start: bool = False
    return_structured: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MFGSolverConfig":
        """Create configuration from dictionary."""
        # Handle nested dataclasses
        if "picard" in config_dict and isinstance(config_dict["picard"], dict):
            config_dict["picard"] = PicardConfig(**config_dict["picard"])
        if "hjb" in config_dict and isinstance(config_dict["hjb"], dict):
            hjb_dict = config_dict["hjb"]
            if "newton" in hjb_dict and isinstance(hjb_dict["newton"], dict):
                hjb_dict["newton"] = NewtonConfig(**hjb_dict["newton"])
            if "gfdm" in hjb_dict and isinstance(hjb_dict["gfdm"], dict):
                hjb_dict["gfdm"] = GFDMConfig(**hjb_dict["gfdm"])
            config_dict["hjb"] = HJBConfig(**hjb_dict)
        if "fp" in config_dict and isinstance(config_dict["fp"], dict):
            fp_dict = config_dict["fp"]
            if "particle" in fp_dict and isinstance(fp_dict["particle"], dict):
                fp_dict["particle"] = ParticleConfig(**fp_dict["particle"])
            config_dict["fp"] = FPConfig(**fp_dict)

        return cls(**config_dict)


# Factory functions for common configurations


def create_default_config() -> MFGSolverConfig:
    """Create default balanced configuration."""
    return MFGSolverConfig()


def create_fast_config() -> MFGSolverConfig:
    """Create configuration optimized for speed."""
    return MFGSolverConfig(
        picard=PicardConfig.fast(),
        hjb=HJBConfig.fast(),
        fp=FPConfig.fast(),
        return_structured=True,  # Structured results are helpful for monitoring
    )


def create_accurate_config() -> MFGSolverConfig:
    """Create configuration optimized for accuracy."""
    return MFGSolverConfig(
        picard=PicardConfig.accurate(),
        hjb=HJBConfig.accurate(),
        fp=FPConfig.accurate(),
        warm_start=True,  # Warm start can help with accuracy in parameter studies
        return_structured=True,
    )


def create_research_config() -> MFGSolverConfig:
    """Create configuration suitable for research with detailed monitoring."""
    config = create_accurate_config()
    config.picard.verbose = True
    config.hjb.newton.verbose = True
    config.metadata = {
        "purpose": "research",
        "monitoring_level": "detailed",
        "created_by": "create_research_config",
    }
    return config


def create_production_config() -> MFGSolverConfig:
    """Create configuration suitable for production use."""
    config = create_fast_config()
    config.picard.verbose = False
    config.hjb.newton.verbose = False
    config.metadata = {
        "purpose": "production",
        "monitoring_level": "minimal",
        "created_by": "create_production_config",
    }
    return config


# Backward compatibility functions for existing parameter handling


def extract_legacy_parameters(config: MFGSolverConfig, **kwargs) -> Dict[str, Any]:
    """
    Extract parameters from legacy keyword arguments and merge with config.

    This function helps maintain backward compatibility while transitioning
    to the new configuration system.
    """
    # Extract Picard parameters
    picard_params = {}
    if "max_iterations" in kwargs:
        picard_params["max_iterations"] = kwargs.pop("max_iterations")
    elif "max_picard_iterations" in kwargs:
        picard_params["max_iterations"] = kwargs.pop("max_picard_iterations")
    elif "Niter_max" in kwargs:
        warnings.warn(
            "Parameter 'Niter_max' is deprecated. Use configuration objects instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        picard_params["max_iterations"] = kwargs.pop("Niter_max")

    if "tolerance" in kwargs:
        picard_params["tolerance"] = kwargs.pop("tolerance")
    elif "picard_tolerance" in kwargs:
        picard_params["tolerance"] = kwargs.pop("picard_tolerance")
    elif "l2errBoundPicard" in kwargs:
        warnings.warn(
            "Parameter 'l2errBoundPicard' is deprecated. Use configuration objects instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        picard_params["tolerance"] = kwargs.pop("l2errBoundPicard")

    # Extract Newton parameters
    newton_params = {}
    if "max_newton_iterations" in kwargs:
        newton_params["max_iterations"] = kwargs.pop("max_newton_iterations")
    elif "NiterNewton" in kwargs:
        warnings.warn(
            "Parameter 'NiterNewton' is deprecated. Use configuration objects instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        newton_params["max_iterations"] = kwargs.pop("NiterNewton")

    if "newton_tolerance" in kwargs:
        newton_params["tolerance"] = kwargs.pop("newton_tolerance")
    elif "l2errBoundNewton" in kwargs:
        warnings.warn(
            "Parameter 'l2errBoundNewton' is deprecated. Use configuration objects instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        newton_params["tolerance"] = kwargs.pop("l2errBoundNewton")

    # Update config with extracted parameters
    if picard_params:
        for key, value in picard_params.items():
            setattr(config.picard, key, value)

    if newton_params:
        for key, value in newton_params.items():
            setattr(config.hjb.newton, key, value)

    # Handle return format preference
    if "return_structured" in kwargs:
        config.return_structured = kwargs.pop("return_structured")

    return kwargs  # Return remaining kwargs
