"""
Plugin system for MFG_PDE enabling third-party solver extensions.

This module provides a flexible plugin architecture that allows researchers
and developers to extend the MFG_PDE framework with custom solvers, algorithms,
and analysis tools without modifying the core codebase.

Features:
- Automatic plugin discovery and registration
- Version compatibility checking
- Dependency validation
- Plugin lifecycle management
- Configuration integration
"""

import importlib
import inspect
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import pkg_resources

from ..config.pydantic_config import MFGSolverConfig

logger = logging.getLogger(__name__)


class PluginStatus(Enum):
    """Plugin status enumeration."""

    DISCOVERED = "discovered"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class PluginMetadata:
    """Plugin metadata and information."""

    name: str
    version: str
    description: str
    author: str
    email: str
    license: str
    homepage: str
    min_mfg_version: str
    max_mfg_version: Optional[str] = None
    dependencies: List[str] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.tags is None:
            self.tags = []


class SolverPlugin(ABC):
    """Abstract base class for MFG solver plugins.

    All plugin developers should inherit from this class and implement
    the required methods to provide custom solver functionality.
    """

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata information."""
        pass

    @abstractmethod
    def get_solver_types(self) -> List[str]:
        """Return list of solver types this plugin provides.

        Returns:
            List of solver type names that can be created by this plugin.
        """
        pass

    @abstractmethod
    def create_solver(
        self,
        problem,
        solver_type: str,
        config: Optional[MFGSolverConfig] = None,
        **kwargs,
    ):
        """Create solver instance for the specified type.

        Args:
            problem: MFG problem instance
            solver_type: Type of solver to create
            config: Optional solver configuration
            **kwargs: Additional solver parameters

        Returns:
            Solver instance implementing the standard MFG solver interface
        """
        pass

    @abstractmethod
    def validate_solver_type(self, solver_type: str) -> bool:
        """Validate if the solver type is supported by this plugin.

        Args:
            solver_type: Solver type to validate

        Returns:
            True if solver type is supported, False otherwise
        """
        pass

    def get_solver_description(self, solver_type: str) -> str:
        """Get description for a specific solver type.

        Args:
            solver_type: Solver type to describe

        Returns:
            Human-readable description of the solver
        """
        return f"Solver of type '{solver_type}' provided by {self.metadata.name}"

    def get_solver_parameters(self, solver_type: str) -> Dict[str, Any]:
        """Get available parameters for a specific solver type.

        Args:
            solver_type: Solver type to get parameters for

        Returns:
            Dictionary describing available parameters
        """
        return {}

    def on_load(self):
        """Called when plugin is loaded. Override for initialization."""
        pass

    def on_unload(self):
        """Called when plugin is unloaded. Override for cleanup."""
        pass


class AnalysisPlugin(ABC):
    """Abstract base class for analysis and post-processing plugins."""

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata information."""
        pass

    @abstractmethod
    def get_analysis_types(self) -> List[str]:
        """Return list of analysis types this plugin provides."""
        pass

    @abstractmethod
    def run_analysis(self, result, analysis_type: str, **kwargs):
        """Run analysis on solver result.

        Args:
            result: Solver result to analyze
            analysis_type: Type of analysis to perform
            **kwargs: Analysis parameters

        Returns:
            Analysis result
        """
        pass


@dataclass
class RegisteredPlugin:
    """Information about a registered plugin."""

    plugin_class: Type[SolverPlugin]
    plugin_instance: Optional[SolverPlugin]
    metadata: PluginMetadata
    status: PluginStatus
    error_message: Optional[str] = None
    load_time: Optional[float] = None


class PluginManager:
    """Central plugin management system for MFG_PDE.

    Handles plugin discovery, registration, loading, and lifecycle management.
    Provides a unified interface for accessing both core and plugin-provided
    solver implementations.
    """

    def __init__(self):
        self.plugins: Dict[str, RegisteredPlugin] = {}
        self.solver_type_registry: Dict[str, str] = {}  # solver_type -> plugin_name
        self.analysis_type_registry: Dict[str, str] = {}  # analysis_type -> plugin_name
        self._core_solvers = self._discover_core_solvers()

        # Initialize logging
        self.logger = logging.getLogger(__name__ + ".PluginManager")

    def discover_plugins(self, search_paths: Optional[List[Path]] = None) -> List[str]:
        """Discover available plugins in specified paths.

        Args:
            search_paths: Optional list of paths to search for plugins.
                         If None, uses default plugin discovery locations.

        Returns:
            List of discovered plugin names
        """
        discovered = []

        if search_paths is None:
            search_paths = self._get_default_plugin_paths()

        # Discover plugins via entry points
        discovered.extend(self._discover_entry_point_plugins())

        # Discover plugins in filesystem paths
        for path in search_paths:
            discovered.extend(self._discover_path_plugins(path))

        self.logger.info(f"Discovered {len(discovered)} plugins")
        return discovered

    def register_plugin(self, plugin_class: Type[SolverPlugin]) -> bool:
        """Register a plugin class.

        Args:
            plugin_class: Plugin class to register

        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Validate plugin class
            if not issubclass(plugin_class, (SolverPlugin, AnalysisPlugin)):
                raise ValueError("Plugin must inherit from SolverPlugin or AnalysisPlugin")

            # Get metadata
            temp_instance = plugin_class()
            metadata = temp_instance.metadata

            # Check version compatibility
            if not self._check_version_compatibility(metadata):
                self.logger.warning(f"Plugin {metadata.name} version incompatible with current MFG_PDE version")
                return False

            # Register plugin
            registered_plugin = RegisteredPlugin(
                plugin_class=plugin_class,
                plugin_instance=None,
                metadata=metadata,
                status=PluginStatus.DISCOVERED,
            )

            self.plugins[metadata.name] = registered_plugin
            self.logger.info(f"Registered plugin: {metadata.name} v{metadata.version}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to register plugin {plugin_class.__name__}: {e}")
            return False

    def load_plugin(self, plugin_name: str) -> bool:
        """Load a registered plugin.

        Args:
            plugin_name: Name of plugin to load

        Returns:
            True if loading successful, False otherwise
        """
        if plugin_name not in self.plugins:
            self.logger.error(f"Plugin {plugin_name} not registered")
            return False

        plugin_info = self.plugins[plugin_name]

        try:
            # Create plugin instance
            plugin_instance = plugin_info.plugin_class()

            # Check dependencies
            if not self._check_dependencies(plugin_instance.metadata):
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_message = "Dependency check failed"
                return False

            # Call plugin initialization
            plugin_instance.on_load()

            # Update plugin info
            plugin_info.plugin_instance = plugin_instance
            plugin_info.status = PluginStatus.LOADED

            # Register solver types
            if isinstance(plugin_instance, SolverPlugin):
                for solver_type in plugin_instance.get_solver_types():
                    self.solver_type_registry[solver_type] = plugin_name

            # Register analysis types
            if isinstance(plugin_instance, AnalysisPlugin):
                for analysis_type in plugin_instance.get_analysis_types():
                    self.analysis_type_registry[analysis_type] = plugin_name

            self.logger.info(f"Loaded plugin: {plugin_name}")
            return True

        except Exception as e:
            plugin_info.status = PluginStatus.ERROR
            plugin_info.error_message = str(e)
            self.logger.error(f"Failed to load plugin {plugin_name}: {e}")
            return False

    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a loaded plugin.

        Args:
            plugin_name: Name of plugin to unload

        Returns:
            True if unloading successful, False otherwise
        """
        if plugin_name not in self.plugins:
            return False

        plugin_info = self.plugins[plugin_name]

        if plugin_info.status != PluginStatus.LOADED:
            return False

        try:
            # Call plugin cleanup
            if plugin_info.plugin_instance:
                plugin_info.plugin_instance.on_unload()

            # Unregister solver types
            if isinstance(plugin_info.plugin_instance, SolverPlugin):
                for solver_type in plugin_info.plugin_instance.get_solver_types():
                    if solver_type in self.solver_type_registry:
                        del self.solver_type_registry[solver_type]

            # Unregister analysis types
            if isinstance(plugin_info.plugin_instance, AnalysisPlugin):
                for analysis_type in plugin_info.plugin_instance.get_analysis_types():
                    if analysis_type in self.analysis_type_registry:
                        del self.analysis_type_registry[analysis_type]

            # Update plugin info
            plugin_info.plugin_instance = None
            plugin_info.status = PluginStatus.DISCOVERED

            self.logger.info(f"Unloaded plugin: {plugin_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False

    def load_all_plugins(self) -> Dict[str, bool]:
        """Load all discovered plugins.

        Returns:
            Dictionary mapping plugin names to load success status
        """
        results = {}
        for plugin_name in self.plugins:
            results[plugin_name] = self.load_plugin(plugin_name)
        return results

    def create_solver(
        self,
        problem,
        solver_type: str,
        config: Optional[MFGSolverConfig] = None,
        **kwargs,
    ):
        """Create solver instance using registered plugins or core solvers.

        Args:
            problem: MFG problem instance
            solver_type: Type of solver to create
            config: Optional solver configuration
            **kwargs: Additional solver parameters

        Returns:
            Solver instance

        Raises:
            ValueError: If solver type is not supported
        """
        # Check if it's a plugin-provided solver
        if solver_type in self.solver_type_registry:
            plugin_name = self.solver_type_registry[solver_type]
            plugin_info = self.plugins[plugin_name]

            if plugin_info.status == PluginStatus.LOADED and plugin_info.plugin_instance:
                return plugin_info.plugin_instance.create_solver(problem, solver_type, config, **kwargs)
            else:
                raise ValueError(f"Plugin {plugin_name} providing {solver_type} is not loaded")

        # Check if it's a core solver
        if solver_type in self._core_solvers:
            return self._create_core_solver(problem, solver_type, config, **kwargs)

        raise ValueError(f"Unknown solver type: {solver_type}")

    def run_analysis(self, result, analysis_type: str, **kwargs):
        """Run analysis using registered analysis plugins.

        Args:
            result: Solver result to analyze
            analysis_type: Type of analysis to perform
            **kwargs: Analysis parameters

        Returns:
            Analysis result

        Raises:
            ValueError: If analysis type is not supported
        """
        if analysis_type not in self.analysis_type_registry:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        plugin_name = self.analysis_type_registry[analysis_type]
        plugin_info = self.plugins[plugin_name]

        if plugin_info.status == PluginStatus.LOADED and plugin_info.plugin_instance:
            return plugin_info.plugin_instance.run_analysis(result, analysis_type, **kwargs)
        else:
            raise ValueError(f"Plugin {plugin_name} providing {analysis_type} is not loaded")

    def list_available_solvers(self) -> Dict[str, Dict[str, Any]]:
        """List all available solver types with descriptions.

        Returns:
            Dictionary mapping solver types to their information
        """
        solvers = {}

        # Add core solvers
        for solver_type in self._core_solvers:
            solvers[solver_type] = {
                "provider": "core",
                "description": f"Core MFG_PDE solver: {solver_type}",
                "parameters": {},
            }

        # Add plugin solvers
        for solver_type, plugin_name in self.solver_type_registry.items():
            plugin_info = self.plugins[plugin_name]
            if plugin_info.status == PluginStatus.LOADED and plugin_info.plugin_instance:
                solvers[solver_type] = {
                    "provider": f"plugin:{plugin_name}",
                    "description": plugin_info.plugin_instance.get_solver_description(solver_type),
                    "parameters": plugin_info.plugin_instance.get_solver_parameters(solver_type),
                }

        return solvers

    def list_available_analyses(self) -> Dict[str, Dict[str, Any]]:
        """List all available analysis types with descriptions.

        Returns:
            Dictionary mapping analysis types to their information
        """
        analyses = {}

        for analysis_type, plugin_name in self.analysis_type_registry.items():
            plugin_info = self.plugins[plugin_name]
            if plugin_info.status == PluginStatus.LOADED and plugin_info.plugin_instance:
                analyses[analysis_type] = {
                    "provider": f"plugin:{plugin_name}",
                    "plugin_metadata": plugin_info.metadata,
                }

        return analyses

    def get_plugin_info(self, plugin_name: str) -> Optional[RegisteredPlugin]:
        """Get information about a specific plugin.

        Args:
            plugin_name: Name of plugin

        Returns:
            Plugin information or None if not found
        """
        return self.plugins.get(plugin_name)

    def list_plugins(self) -> Dict[str, RegisteredPlugin]:
        """List all registered plugins.

        Returns:
            Dictionary of all registered plugins
        """
        return self.plugins.copy()

    def _discover_core_solvers(self) -> List[str]:
        """Discover core solver types available in MFG_PDE."""
        # This would scan the core solver modules to find available types
        # For now, return the known core solver types
        return ["fixed_point", "particle_collocation", "hjb_fdm", "hjb_collocation"]

    def _create_core_solver(
        self,
        problem,
        solver_type: str,
        config: Optional[MFGSolverConfig] = None,
        **kwargs,
    ):
        """Create core solver instance."""
        # Import here to avoid circular imports
        from ..factory import create_solver

        return create_solver(problem, solver_type, config=config, **kwargs)

    def _get_default_plugin_paths(self) -> List[Path]:
        """Get default plugin search paths."""
        paths = []

        # User plugin directory
        home = Path.home()
        paths.append(home / ".mfg_pde" / "plugins")

        # System plugin directory
        import mfg_pde

        package_dir = Path(mfg_pde.__file__).parent
        paths.append(package_dir / "plugins")

        # Current working directory plugins
        paths.append(Path.cwd() / "mfg_plugins")

        return [p for p in paths if p.exists()]

    def _discover_entry_point_plugins(self) -> List[str]:
        """Discover plugins via setuptools entry points."""
        discovered = []

        try:
            for entry_point in pkg_resources.iter_entry_points("mfg_pde.plugins"):
                try:
                    plugin_class = entry_point.load()
                    if self.register_plugin(plugin_class):
                        discovered.append(entry_point.name)
                except Exception as e:
                    self.logger.error(f"Failed to load entry point plugin {entry_point.name}: {e}")
        except Exception as e:
            self.logger.warning(f"Error discovering entry point plugins: {e}")

        return discovered

    def _discover_path_plugins(self, path: Path) -> List[str]:
        """Discover plugins in a filesystem path."""
        discovered = []

        if not path.exists() or not path.is_dir():
            return discovered

        for plugin_file in path.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue

            try:
                # Dynamic import
                spec = importlib.util.spec_from_file_location(plugin_file.stem, plugin_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find plugin classes
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, (SolverPlugin, AnalysisPlugin)) and obj not in (
                        SolverPlugin,
                        AnalysisPlugin,
                    ):
                        if self.register_plugin(obj):
                            discovered.append(name)

            except Exception as e:
                self.logger.error(f"Failed to load plugin file {plugin_file}: {e}")

        return discovered

    def _check_version_compatibility(self, metadata: PluginMetadata) -> bool:
        """Check if plugin is compatible with current MFG_PDE version."""
        import mfg_pde

        current_version = mfg_pde.__version__

        # Simple version checking (could be enhanced with proper semantic versioning)
        if metadata.min_mfg_version and current_version < metadata.min_mfg_version:
            return False

        if metadata.max_mfg_version and current_version > metadata.max_mfg_version:
            return False

        return True

    def _check_dependencies(self, metadata: PluginMetadata) -> bool:
        """Check if plugin dependencies are available."""
        for dependency in metadata.dependencies:
            try:
                importlib.import_module(dependency)
            except ImportError:
                self.logger.error(f"Missing dependency: {dependency}")
                return False
        return True


# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


def register_plugin(plugin_class: Type[SolverPlugin]) -> bool:
    """Convenience function to register a plugin."""
    return get_plugin_manager().register_plugin(plugin_class)


def discover_and_load_plugins(
    search_paths: Optional[List[Path]] = None,
) -> Dict[str, bool]:
    """Convenience function to discover and load all plugins."""
    manager = get_plugin_manager()
    manager.discover_plugins(search_paths)
    return manager.load_all_plugins()


def create_solver_with_plugins(problem, solver_type: str, config: Optional[MFGSolverConfig] = None, **kwargs):
    """Create solver with plugin support."""
    return get_plugin_manager().create_solver(problem, solver_type, config, **kwargs)


def list_available_solvers() -> Dict[str, Dict[str, Any]]:
    """List all available solver types including plugins."""
    return get_plugin_manager().list_available_solvers()
