"""
Hooks System for MFG_PDE

This module provides the hooks pattern for customizing solver behavior
without requiring complex inheritance or configuration.

Basic Usage:
    from mfg_pde.hooks import SolverHooks, DebugHook, PlottingHook

    hooks = DebugHook()
    result = solver.solve(problem, hooks=hooks)

Advanced Usage:
    from mfg_pde.hooks import MultiHook

    combined = MultiHook(DebugHook(), PlottingHook(), CustomHook())
    result = solver.solve(problem, hooks=combined)
"""

from .base import SolverHooks
from .composition import ChainHook, ConditionalHook, FilterHook, MultiHook, PriorityHook, TransformHook

# Control flow hooks are imported separately to avoid heavy dependencies
try:
    from .control_flow import (
        AdaptiveControlHook,  # noqa: F401
        ConditionalStopHook,  # noqa: F401
        ControlState,  # noqa: F401
        PerformanceControlHook,  # noqa: F401
        WatchdogHook,  # noqa: F401
    )

    _CONTROL_FLOW_AVAILABLE = True
except ImportError:
    _CONTROL_FLOW_AVAILABLE = False

__all__ = ["ChainHook", "ConditionalHook", "FilterHook", "MultiHook", "PriorityHook", "SolverHooks", "TransformHook"]

# Add control flow exports if available
if _CONTROL_FLOW_AVAILABLE:
    __all__.extend(
        ["AdaptiveControlHook", "ConditionalStopHook", "ControlState", "PerformanceControlHook", "WatchdogHook"]
    )

# Import debugging and visualization hooks with optional dependencies
try:
    from .debug import ConvergenceAnalysisHook, DebugHook, PerformanceHook, StateInspectionHook  # noqa: F401

    _DEBUG_AVAILABLE = True
except ImportError:
    _DEBUG_AVAILABLE = False

try:
    from .visualization import AnimationHook, LoggingHook, PlottingHook, ProgressBarHook  # noqa: F401

    _VISUALIZATION_AVAILABLE = True
except ImportError:
    _VISUALIZATION_AVAILABLE = False

# Add debugging exports if available
if _DEBUG_AVAILABLE:
    __all__.extend(["ConvergenceAnalysisHook", "DebugHook", "PerformanceHook", "StateInspectionHook"])

# Add visualization exports if available
if _VISUALIZATION_AVAILABLE:
    __all__.extend(["AnimationHook", "LoggingHook", "PlottingHook", "ProgressBarHook"])

# Import extension points
try:
    from .extensions import (
        AdaptiveParameterHook,  # noqa: F401
        AlgorithmExtensionHook,  # noqa: F401
        CustomConvergenceHook,  # noqa: F401
        CustomFPHook,  # noqa: F401
        CustomHJBHook,  # noqa: F401
        CustomInitializationHook,  # noqa: F401
        CustomResidualHook,  # noqa: F401
        MethodSwitchingHook,  # noqa: F401
        PostprocessingHook,  # noqa: F401
        PreprocessingHook,  # noqa: F401
    )

    _EXTENSIONS_AVAILABLE = True
except ImportError:
    _EXTENSIONS_AVAILABLE = False

# Add extension exports if available
if _EXTENSIONS_AVAILABLE:
    __all__.extend(
        [
            "AdaptiveParameterHook",
            "AlgorithmExtensionHook",
            "CustomConvergenceHook",
            "CustomFPHook",
            "CustomHJBHook",
            "CustomInitializationHook",
            "CustomResidualHook",
            "MethodSwitchingHook",
            "PostprocessingHook",
            "PreprocessingHook",
        ]
    )
