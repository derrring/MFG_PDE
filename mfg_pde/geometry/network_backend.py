"""
Unified Network Backend System for MFG_PDE.

This module provides a unified interface for multiple network libraries:
- igraph: Primary backend (C-based, fast, good balance)
- networkit: Large-scale networks (parallel, billions of nodes)
- networkx: Algorithm completeness (fallback, most comprehensive)

The backend system automatically selects the optimal library based on:
- Network size
- Operation type  
- Available libraries
- Performance requirements
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix


class NetworkBackendType(Enum):
    """Available network backend types."""

    IGRAPH = "igraph"
    NETWORKIT = "networkit"
    NETWORKX = "networkx"
    AUTO = "auto"


class OperationType(Enum):
    """Types of network operations for backend optimization."""

    GENERAL = "general"
    LARGE_SCALE = "large_scale"
    ALGORITHMS = "algorithms"
    VISUALIZATION = "visualization"
    ANALYSIS = "analysis"
    SHORTEST_PATHS = "shortest_paths"
    CENTRALITY = "centrality"
    COMMUNITY = "community"


@dataclass
class BackendCapabilities:
    """Capabilities and performance characteristics of a backend."""

    max_recommended_nodes: int
    max_theoretical_nodes: int
    parallel_support: bool
    visualization_quality: int  # 1-5 scale
    algorithm_coverage: int  # 1-5 scale
    memory_efficiency: int  # 1-5 scale
    speed_rating: int  # 1-5 scale
    installation_difficulty: int  # 1-5 scale (1=easy)


class NetworkBackendError(Exception):
    """Base exception for network backend issues."""

    pass


class BackendNotAvailableError(NetworkBackendError):
    """Raised when requested backend is not available."""

    pass


class AbstractNetworkBackend(ABC):
    """Abstract base class for network backends."""

    @abstractmethod
    def create_graph(self, num_nodes: int, **kwargs) -> Any:
        """Create empty graph with specified number of nodes."""
        pass

    @abstractmethod
    def add_edges(
        self,
        graph: Any,
        edges: List[Tuple[int, int]],
        weights: Optional[np.ndarray] = None,
    ) -> Any:
        """Add edges to graph."""
        pass

    @abstractmethod
    def get_adjacency_matrix(self, graph: Any) -> csr_matrix:
        """Get sparse adjacency matrix."""
        pass

    @abstractmethod
    def get_laplacian_matrix(self, graph: Any) -> csr_matrix:
        """Get graph Laplacian matrix."""
        pass

    @abstractmethod
    def shortest_paths(self, graph: Any, source: Optional[int] = None) -> np.ndarray:
        """Compute shortest paths."""
        pass

    @abstractmethod
    def connected_components(self, graph: Any) -> List[List[int]]:
        """Find connected components."""
        pass

    @abstractmethod
    def node_degrees(self, graph: Any) -> np.ndarray:
        """Get node degrees."""
        pass

    @abstractmethod
    def clustering_coefficient(self, graph: Any) -> float:
        """Compute average clustering coefficient."""
        pass


class IGraphBackend(AbstractNetworkBackend):
    """igraph backend implementation."""

    def __init__(self):
        try:
            import igraph as ig

            self.ig = ig
            self.available = True
        except ImportError:
            self.ig = None
            self.available = False

    def create_graph(self, num_nodes: int, directed: bool = False, **kwargs) -> Any:
        if not self.available:
            raise BackendNotAvailableError("igraph not available")
        return self.ig.Graph(n=num_nodes, directed=directed)

    def add_edges(
        self,
        graph: Any,
        edges: List[Tuple[int, int]],
        weights: Optional[np.ndarray] = None,
    ) -> Any:
        graph.add_edges(edges)
        if weights is not None:
            graph.es["weight"] = weights
        return graph

    def get_adjacency_matrix(self, graph: Any) -> csr_matrix:
        # igraph returns dense by default, convert to sparse
        adj_dense = np.array(graph.get_adjacency(attribute="weight").data)
        return csr_matrix(adj_dense)

    def get_laplacian_matrix(self, graph: Any) -> csr_matrix:
        # Compute Laplacian as D - A
        adj = self.get_adjacency_matrix(graph)
        degrees = np.array(adj.sum(axis=1)).flatten()
        from scipy.sparse import diags

        degree_matrix = diags(degrees, format="csr")
        return degree_matrix - adj

    def shortest_paths(self, graph: Any, source: Optional[int] = None) -> np.ndarray:
        if source is not None:
            paths = graph.shortest_paths_dijkstra(source=source, weights="weight")
            return np.array(paths[0])
        else:
            paths = graph.shortest_paths_dijkstra(weights="weight")
            return np.array(paths)

    def connected_components(self, graph: Any) -> List[List[int]]:
        components = graph.connected_components()
        return [list(component) for component in components]

    def node_degrees(self, graph: Any) -> np.ndarray:
        return np.array(graph.degree())

    def clustering_coefficient(self, graph: Any) -> float:
        try:
            clustering = graph.transitivity_avglocal_undirected()
            return clustering if clustering is not None else 0.0
        except:
            return 0.0


class NetworkitBackend(AbstractNetworkBackend):
    """networkit backend implementation."""

    def __init__(self):
        try:
            import networkit as nk

            self.nk = nk
            self.available = True
        except ImportError:
            self.nk = None
            self.available = False

    def create_graph(
        self, num_nodes: int, directed: bool = False, weighted: bool = True, **kwargs
    ) -> Any:
        if not self.available:
            raise BackendNotAvailableError("networkit not available")
        return self.nk.Graph(n=num_nodes, weighted=weighted, directed=directed)

    def add_edges(
        self,
        graph: Any,
        edges: List[Tuple[int, int]],
        weights: Optional[np.ndarray] = None,
    ) -> Any:
        for i, (u, v) in enumerate(edges):
            weight = weights[i] if weights is not None else 1.0
            graph.addEdge(u, v, weight)
        return graph

    def get_adjacency_matrix(self, graph: Any) -> csr_matrix:
        # Convert networkit graph to scipy sparse matrix
        num_nodes = graph.numberOfNodes()
        rows, cols, data = [], [], []

        for u in range(num_nodes):
            for v in graph.iterNeighbors(u):
                rows.append(u)
                cols.append(v)
                data.append(graph.weight(u, v) if graph.isWeighted() else 1.0)

        return csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))

    def get_laplacian_matrix(self, graph: Any) -> csr_matrix:
        adj = self.get_adjacency_matrix(graph)
        degrees = np.array(adj.sum(axis=1)).flatten()
        from scipy.sparse import diags

        degree_matrix = diags(degrees, format="csr")
        return degree_matrix - adj

    def shortest_paths(self, graph: Any, source: Optional[int] = None) -> np.ndarray:
        if source is not None:
            spsp = self.nk.distance.Dijkstra(graph, source)
            spsp.run()
            return np.array(spsp.getDistances())
        else:
            apsp = self.nk.distance.APSP(graph)
            apsp.run()
            return np.array(apsp.getDistanceMatrix())

    def connected_components(self, graph: Any) -> List[List[int]]:
        cc = self.nk.components.ConnectedComponents(graph)
        cc.run()
        components = []
        for i in range(cc.numberOfComponents()):
            component = []
            for node in range(graph.numberOfNodes()):
                if cc.componentOfNode(node) == i:
                    component.append(node)
            components.append(component)
        return components

    def node_degrees(self, graph: Any) -> np.ndarray:
        return np.array([graph.degree(node) for node in range(graph.numberOfNodes())])

    def clustering_coefficient(self, graph: Any) -> float:
        try:
            cc = self.nk.centrality.LocalClusteringCoefficient(graph)
            cc.run()
            coefficients = cc.scores()
            return np.mean(coefficients) if coefficients else 0.0
        except:
            return 0.0


class NetworkXBackend(AbstractNetworkBackend):
    """networkx backend implementation."""

    def __init__(self):
        try:
            import networkx as nx

            self.nx = nx
            self.available = True
        except ImportError:
            self.nx = None
            self.available = False

    def create_graph(self, num_nodes: int, directed: bool = False, **kwargs) -> Any:
        if not self.available:
            raise BackendNotAvailableError("networkx not available")

        if directed:
            graph = self.nx.DiGraph()
        else:
            graph = self.nx.Graph()

        graph.add_nodes_from(range(num_nodes))
        return graph

    def add_edges(
        self,
        graph: Any,
        edges: List[Tuple[int, int]],
        weights: Optional[np.ndarray] = None,
    ) -> Any:
        if weights is not None:
            weighted_edges = [
                (u, v, {"weight": w}) for (u, v), w in zip(edges, weights)
            ]
            graph.add_edges_from(weighted_edges)
        else:
            graph.add_edges_from(edges)
        return graph

    def get_adjacency_matrix(self, graph: Any) -> csr_matrix:
        return self.nx.adjacency_matrix(graph, weight="weight")

    def get_laplacian_matrix(self, graph: Any) -> csr_matrix:
        return self.nx.laplacian_matrix(graph, weight="weight")

    def shortest_paths(self, graph: Any, source: Optional[int] = None) -> np.ndarray:
        if source is not None:
            paths = self.nx.single_source_dijkstra_path_length(
                graph, source, weight="weight"
            )
            result = np.full(graph.number_of_nodes(), np.inf)
            for target, length in paths.items():
                result[target] = length
            return result
        else:
            paths = dict(self.nx.all_pairs_dijkstra_path_length(graph, weight="weight"))
            result = np.full((graph.number_of_nodes(), graph.number_of_nodes()), np.inf)
            for source, targets in paths.items():
                for target, length in targets.items():
                    result[source, target] = length
            return result

    def connected_components(self, graph: Any) -> List[List[int]]:
        if graph.is_directed():
            components = self.nx.strongly_connected_components(graph)
        else:
            components = self.nx.connected_components(graph)
        return [list(component) for component in components]

    def node_degrees(self, graph: Any) -> np.ndarray:
        degrees = dict(graph.degree(weight="weight"))
        return np.array([degrees[node] for node in sorted(degrees.keys())])

    def clustering_coefficient(self, graph: Any) -> float:
        return self.nx.average_clustering(graph, weight="weight")


class NetworkBackendManager:
    """
    Unified network backend manager.

    Automatically selects optimal backend based on problem size and operation type.
    Provides fallback mechanisms and performance optimization.
    """

    # Backend capabilities database
    BACKEND_CAPABILITIES = {
        NetworkBackendType.IGRAPH: BackendCapabilities(
            max_recommended_nodes=100_000,
            max_theoretical_nodes=1_000_000,
            parallel_support=False,
            visualization_quality=5,
            algorithm_coverage=4,
            memory_efficiency=4,
            speed_rating=4,
            installation_difficulty=2,
        ),
        NetworkBackendType.NETWORKIT: BackendCapabilities(
            max_recommended_nodes=10_000_000,
            max_theoretical_nodes=1_000_000_000,
            parallel_support=True,
            visualization_quality=2,
            algorithm_coverage=3,
            memory_efficiency=5,
            speed_rating=5,
            installation_difficulty=3,
        ),
        NetworkBackendType.NETWORKX: BackendCapabilities(
            max_recommended_nodes=10_000,
            max_theoretical_nodes=100_000,
            parallel_support=False,
            visualization_quality=3,
            algorithm_coverage=5,
            memory_efficiency=2,
            speed_rating=2,
            installation_difficulty=1,
        ),
    }

    def __init__(
        self, preferred_backend: NetworkBackendType = NetworkBackendType.IGRAPH
    ):
        self.preferred_backend = preferred_backend
        self.backends = self._initialize_backends()
        self.available_backends = [
            bt for bt, backend in self.backends.items() if backend.available
        ]

        if not self.available_backends:
            raise NetworkBackendError(
                "No network backends available. Install igraph, networkit, or networkx."
            )

        # Issue warnings for missing recommended backends
        self._check_backend_availability()

    def _initialize_backends(self) -> Dict[NetworkBackendType, AbstractNetworkBackend]:
        """Initialize all available backends."""
        backends = {
            NetworkBackendType.IGRAPH: IGraphBackend(),
            NetworkBackendType.NETWORKIT: NetworkitBackend(),
            NetworkBackendType.NETWORKX: NetworkXBackend(),
        }
        return backends

    def _check_backend_availability(self):
        """Check backend availability and issue helpful warnings."""
        if NetworkBackendType.IGRAPH not in self.available_backends:
            warnings.warn(
                "igraph not available. Install with: pip install igraph\n"
                "igraph provides excellent performance for most network operations.",
                ImportWarning,
            )

        if NetworkBackendType.NETWORKIT not in self.available_backends:
            warnings.warn(
                "networkit not available. Install with: pip install networkit\n"
                "networkit is recommended for large-scale network analysis (>100K nodes).",
                ImportWarning,
            )

        if NetworkBackendType.NETWORKX not in self.available_backends:
            warnings.warn(
                "networkx not available. Install with: pip install networkx\n"
                "networkx provides the most comprehensive algorithm library.",
                ImportWarning,
            )

    def choose_backend(
        self,
        num_nodes: int,
        operation_type: OperationType = OperationType.GENERAL,
        force_backend: Optional[NetworkBackendType] = None,
    ) -> NetworkBackendType:
        """
        Choose optimal backend based on problem characteristics.

        Args:
            num_nodes: Number of nodes in the network
            operation_type: Type of operation to be performed
            force_backend: Force specific backend (if available)

        Returns:
            Optimal backend type
        """
        if force_backend and force_backend in self.available_backends:
            return force_backend

        # Operation-specific preferences
        operation_preferences = {
            OperationType.LARGE_SCALE: [
                NetworkBackendType.NETWORKIT,
                NetworkBackendType.IGRAPH,
            ],
            OperationType.VISUALIZATION: [
                NetworkBackendType.IGRAPH,
                NetworkBackendType.NETWORKX,
            ],
            OperationType.ALGORITHMS: [
                NetworkBackendType.NETWORKX,
                NetworkBackendType.IGRAPH,
            ],
            OperationType.SHORTEST_PATHS: [
                NetworkBackendType.NETWORKIT,
                NetworkBackendType.IGRAPH,
            ],
            OperationType.CENTRALITY: [
                NetworkBackendType.NETWORKIT,
                NetworkBackendType.IGRAPH,
            ],
            OperationType.COMMUNITY: [
                NetworkBackendType.IGRAPH,
                NetworkBackendType.NETWORKX,
            ],
        }

        # Size-based selection
        if num_nodes > 100_000:
            size_preferences = [NetworkBackendType.NETWORKIT, NetworkBackendType.IGRAPH]
        elif num_nodes > 10_000:
            size_preferences = [NetworkBackendType.IGRAPH, NetworkBackendType.NETWORKIT]
        else:
            size_preferences = [NetworkBackendType.IGRAPH, NetworkBackendType.NETWORKX]

        # Combine preferences
        candidates = operation_preferences.get(operation_type, size_preferences)

        # Choose first available backend from preferences
        for backend_type in candidates:
            if backend_type in self.available_backends:
                return backend_type

        # Fallback to any available backend
        if self.preferred_backend in self.available_backends:
            return self.preferred_backend

        return self.available_backends[0]

    def get_backend(self, backend_type: NetworkBackendType) -> AbstractNetworkBackend:
        """Get backend instance."""
        if backend_type not in self.available_backends:
            raise BackendNotAvailableError(
                f"Backend {backend_type.value} not available"
            )
        return self.backends[backend_type]

    def create_optimized_graph(
        self,
        num_nodes: int,
        operation_type: OperationType = OperationType.GENERAL,
        force_backend: Optional[NetworkBackendType] = None,
        **kwargs,
    ) -> Tuple[Any, NetworkBackendType]:
        """
        Create graph using optimal backend.

        Returns:
            (graph_object, backend_type_used)
        """
        backend_type = self.choose_backend(num_nodes, operation_type, force_backend)
        backend = self.get_backend(backend_type)
        graph = backend.create_graph(num_nodes, **kwargs)
        return graph, backend_type

    def get_capabilities(self, backend_type: NetworkBackendType) -> BackendCapabilities:
        """Get capabilities of a specific backend."""
        return self.BACKEND_CAPABILITIES[backend_type]

    def get_backend_info(self) -> Dict[str, Any]:
        """Get comprehensive backend information."""
        info = {
            "available_backends": [bt.value for bt in self.available_backends],
            "preferred_backend": self.preferred_backend.value,
            "backend_capabilities": {},
        }

        for backend_type in self.available_backends:
            caps = self.get_capabilities(backend_type)
            info["backend_capabilities"][backend_type.value] = {
                "max_recommended_nodes": caps.max_recommended_nodes,
                "parallel_support": caps.parallel_support,
                "speed_rating": caps.speed_rating,
                "algorithm_coverage": caps.algorithm_coverage,
            }

        return info


# Global backend manager instance
_backend_manager = None


def get_backend_manager(
    preferred_backend: NetworkBackendType = NetworkBackendType.IGRAPH,
) -> NetworkBackendManager:
    """Get global backend manager instance."""
    global _backend_manager
    if _backend_manager is None:
        _backend_manager = NetworkBackendManager(preferred_backend)
    return _backend_manager


def set_preferred_backend(backend_type: NetworkBackendType):
    """Set preferred backend globally."""
    global _backend_manager
    _backend_manager = NetworkBackendManager(backend_type)
