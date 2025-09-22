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
from .composition import (
    MultiHook, ConditionalHook, PriorityHook,
    FilterHook, TransformHook, ChainHook
)

# Control flow hooks are imported separately to avoid heavy dependencies
try:
    from .control_flow import (
        ControlState, AdaptiveControlHook, PerformanceControlHook,
        WatchdogHook, ConditionalStopHook
    )
    _CONTROL_FLOW_AVAILABLE = True
except ImportError:
    _CONTROL_FLOW_AVAILABLE = False

__all__ = [
    'SolverHooks',
    'MultiHook',
    'ConditionalHook',
    'PriorityHook',
    'FilterHook',
    'TransformHook',
    'ChainHook'
]

# Add control flow exports if available
if _CONTROL_FLOW_AVAILABLE:
    __all__.extend([
        'ControlState',
        'AdaptiveControlHook',
        'PerformanceControlHook',
        'WatchdogHook',
        'ConditionalStopHook'
    ])

# Import debugging and visualization hooks with optional dependencies
try:
    from .debug import (
        DebugHook, PerformanceHook, ConvergenceAnalysisHook,
        StateInspectionHook
    )
    _DEBUG_AVAILABLE = True
except ImportError:
    _DEBUG_AVAILABLE = False

try:
    from .visualization import (
        PlottingHook, AnimationHook, LoggingHook, ProgressBarHook
    )
    _VISUALIZATION_AVAILABLE = True
except ImportError:
    _VISUALIZATION_AVAILABLE = False

# Add debugging exports if available
if _DEBUG_AVAILABLE:
    __all__.extend([
        'DebugHook',
        'PerformanceHook',
        'ConvergenceAnalysisHook',
        'StateInspectionHook'
    ])

# Add visualization exports if available
if _VISUALIZATION_AVAILABLE:
    __all__.extend([
        'PlottingHook',
        'AnimationHook',
        'LoggingHook',
        'ProgressBarHook'
    ])

# Import extension points
try:
    from .extensions import (
        AlgorithmExtensionHook, CustomHJBHook, CustomFPHook,
        CustomConvergenceHook, PreprocessingHook, PostprocessingHook,
        CustomResidualHook, AdaptiveParameterHook, MethodSwitchingHook,
        CustomInitializationHook
    )
    _EXTENSIONS_AVAILABLE = True
except ImportError:
    _EXTENSIONS_AVAILABLE = False

# Add extension exports if available
if _EXTENSIONS_AVAILABLE:
    __all__.extend([
        'AlgorithmExtensionHook',
        'CustomHJBHook',
        'CustomFPHook',
        'CustomConvergenceHook',
        'PreprocessingHook',
        'PostprocessingHook',
        'CustomResidualHook',
        'AdaptiveParameterHook',
        'MethodSwitchingHook',
        'CustomInitializationHook'
    ])