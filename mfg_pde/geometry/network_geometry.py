"""
Network Geometry for Mean Field Games on Graphs.

This module implements geometric representations for MFG problems on network/graph structures,
extending the MFG framework from continuous domains to discrete network topologies.

Key concepts:
- Nodes: Discrete locations where players can position themselves
- Edges: Connections between nodes allowing player movement  
- Network flows: Discrete analogue of continuous density evolution
- Graph Laplacians: Discrete operators for diffusion and gradient terms
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, diags
from scipy.spatial.distance import pdist, squareform

# Import unified backend system
from .network_backend import (
    BackendNotAvailableError,
    get_backend_manager,
    NetworkBackendManager,
    NetworkBackendType,
    OperationType,
)

# Legacy compatibility - keep old imports for fallback
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

try:
    import igraph as ig

    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False
    ig = None


class NetworkType(Enum):
    """Types of network structures for MFG problems."""

    GRID = "grid"
    RANDOM = "random"
    SCALE_FREE = "scale_free"
    SMALL_WORLD = "small_world"
    COMPLETE = "complete"
    TREE = "tree"
    LATTICE = "lattice"
    CUSTOM = "custom"


@dataclass
class NetworkData:
    """
    Network data container for MFG on graphs.

    This class stores all geometric and topological information needed
    to solve MFG problems on network structures.

    Attributes:
        adjacency_matrix: (N, N) sparse adjacency matrix
        node_positions: (N, d) coordinates for visualization [optional]
        edge_weights: Edge weight vector [optional, defaults to 1]
        node_weights: Node weight vector [optional, defaults to 1]
        laplacian_matrix: Graph Laplacian for diffusion operators
        degree_matrix: Node degree matrix
        incidence_matrix: Node-edge incidence matrix
        num_nodes: Number of nodes in network
        num_edges: Number of edges in network
        network_type: Type of network structure
        metadata: Additional network properties
    """

    adjacency_matrix: csr_matrix
    num_nodes: int
    num_edges: int
    network_type: NetworkType = NetworkType.CUSTOM

    # Optional geometric information
    node_positions: Optional[np.ndarray] = None  # (N, d) coordinates
    edge_weights: Optional[np.ndarray] = None  # Edge weights
    node_weights: Optional[np.ndarray] = None  # Node weights/capacities

    # Derived matrices (computed automatically)
    laplacian_matrix: Optional[csr_matrix] = None
    degree_matrix: Optional[csr_matrix] = None
    incidence_matrix: Optional[csr_matrix] = None

    # Network properties
    is_directed: bool = False
    is_weighted: bool = False
    is_connected: bool = True

    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Backend information
    backend_type: Optional[NetworkBackendType] = None
    backend_graph: Optional[Any] = None  # Original backend graph object

    def __post_init__(self):
        """Compute derived network properties and matrices."""
        self._validate_network_data()
        self._compute_derived_matrices()
        self._compute_network_properties()

    def _validate_network_data(self):
        """Validate network data consistency."""
        assert (
            self.adjacency_matrix.shape[0] == self.adjacency_matrix.shape[1]
        ), "Adjacency matrix must be square"
        assert (
            self.adjacency_matrix.shape[0] == self.num_nodes
        ), f"Adjacency matrix size {self.adjacency_matrix.shape[0]} != num_nodes {self.num_nodes}"

        if self.node_positions is not None:
            assert (
                self.node_positions.shape[0] == self.num_nodes
            ), "Node positions must match number of nodes"

        if self.edge_weights is not None:
            assert (
                len(self.edge_weights) == self.num_edges
            ), "Edge weights must match number of edges"

        if self.node_weights is not None:
            assert (
                len(self.node_weights) == self.num_nodes
            ), "Node weights must match number of nodes"

    def _compute_derived_matrices(self):
        """Compute graph Laplacian and other derived matrices."""
        A = self.adjacency_matrix

        # Default edge weights
        if self.edge_weights is None:
            self.edge_weights = np.ones(self.num_edges)

        # Default node weights
        if self.node_weights is None:
            self.node_weights = np.ones(self.num_nodes)

        # Degree matrix
        degrees = np.array(A.sum(axis=1)).flatten()
        self.degree_matrix = diags(degrees, format="csr")

        # Graph Laplacian (L = D - A)
        self.laplacian_matrix = self.degree_matrix - A

        # Incidence matrix (for edge-based operations)
        self.incidence_matrix = self._compute_incidence_matrix()

    def _compute_incidence_matrix(self) -> csr_matrix:
        """Compute node-edge incidence matrix."""
        rows, cols = [], []
        data = []

        edge_idx = 0
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if self.adjacency_matrix[i, j] != 0:
                    # Edge connects nodes i and j
                    rows.extend([i, j])
                    cols.extend([edge_idx, edge_idx])
                    data.extend([1, -1])  # Oriented incidence
                    edge_idx += 1

        return csr_matrix((data, (rows, cols)), shape=(self.num_nodes, self.num_edges))

    def _compute_network_properties(self):
        """Compute basic network properties."""
        A = self.adjacency_matrix

        # Check if directed
        self.is_directed = not (A != A.T).nnz == 0

        # Check if weighted
        self.is_weighted = not np.allclose(A.data, 1.0)

        # Check connectivity (simplified check)
        self.is_connected = self._check_connectivity()

        # Store additional properties
        self.metadata.update(
            {
                "average_degree": np.mean(np.array(A.sum(axis=1)).flatten()),
                "max_degree": np.max(np.array(A.sum(axis=1)).flatten()),
                "density": self.num_edges / (self.num_nodes * (self.num_nodes - 1) / 2),
                "clustering_coefficient": self._compute_clustering_coefficient(),
            }
        )

    def _check_connectivity(self) -> bool:
        """Check if network is connected (simplified version)."""
        if not NETWORKX_AVAILABLE:
            return True  # Assume connected if NetworkX not available

        # Convert to NetworkX for connectivity check
        G = nx.from_scipy_sparse_array(self.adjacency_matrix)
        return (
            nx.is_connected(G) if not self.is_directed else nx.is_strongly_connected(G)
        )

    def _compute_clustering_coefficient(self) -> float:
        """Compute average clustering coefficient."""
        if not NETWORKX_AVAILABLE:
            return 0.0

        G = nx.from_scipy_sparse_array(self.adjacency_matrix)
        return nx.average_clustering(G)

    def get_neighbors(self, node_id: int) -> List[int]:
        """Get neighbors of a given node."""
        return self.adjacency_matrix.getrow(node_id).nonzero()[1].tolist()

    def get_edge_weight(self, node_i: int, node_j: int) -> float:
        """Get weight of edge between two nodes."""
        return self.adjacency_matrix[node_i, node_j]

    def get_node_degree(self, node_id: int) -> int:
        """Get degree of a node."""
        return int(self.adjacency_matrix.getrow(node_id).sum())


class BaseNetworkGeometry(ABC):
    """
    Abstract base class for network geometries in MFG problems.

    This class defines the interface for creating and managing
    network structures for Mean Field Games on graphs.
    """

    def __init__(
        self,
        num_nodes: int,
        network_type: NetworkType = NetworkType.CUSTOM,
        backend_preference: NetworkBackendType = NetworkBackendType.IGRAPH,
    ):
        self.num_nodes = num_nodes
        self.network_type = network_type
        self.network_data: Optional[NetworkData] = None
        self.backend_preference = backend_preference
        self.backend_manager = get_backend_manager(backend_preference)

    @abstractmethod
    def create_network(self, **kwargs) -> NetworkData:
        """Create network with specified parameters."""
        pass

    @abstractmethod
    def compute_distance_matrix(self) -> np.ndarray:
        """Compute shortest path distances between all node pairs."""
        pass

    def get_laplacian_operator(
        self, operator_type: str = "combinatorial"
    ) -> csr_matrix:
        """
        Get graph Laplacian operator for MFG computations.

        Args:
            operator_type: Type of Laplacian ("combinatorial", "normalized", "random_walk")

        Returns:
            Sparse Laplacian matrix
        """
        if self.network_data is None:
            raise ValueError(
                "Network data not initialized. Call create_network() first."
            )

        A = self.network_data.adjacency_matrix
        D = self.network_data.degree_matrix

        if operator_type == "combinatorial":
            return D - A
        elif operator_type == "normalized":
            # L_norm = D^(-1/2) * L * D^(-1/2)
            D_inv_sqrt = diags(1.0 / np.sqrt(D.diagonal() + 1e-12), format="csr")
            L = D - A
            return D_inv_sqrt @ L @ D_inv_sqrt
        elif operator_type == "random_walk":
            # L_rw = D^(-1) * L
            D_inv = diags(1.0 / (D.diagonal() + 1e-12), format="csr")
            L = D - A
            return D_inv @ L
        else:
            raise ValueError(f"Unknown Laplacian type: {operator_type}")

    def _create_network_data_from_backend(
        self,
        backend_graph: Any,
        backend_type: NetworkBackendType,
        node_positions: Optional[np.ndarray] = None,
    ) -> NetworkData:
        """
        Convert backend graph to NetworkData format.

        Args:
            backend_graph: Graph object from backend
            backend_type: Which backend was used
            node_positions: Optional node positions for visualization

        Returns:
            NetworkData object with all matrices computed
        """
        backend = self.backend_manager.get_backend(backend_type)

        # Get adjacency matrix
        adjacency_matrix = backend.get_adjacency_matrix(backend_graph)

        # Count edges (handle directed/undirected)
        if hasattr(backend_graph, "is_directed"):
            is_directed = backend_graph.is_directed()
        else:
            # Check if matrix is symmetric
            is_directed = not (adjacency_matrix != adjacency_matrix.T).nnz == 0

        num_edges = adjacency_matrix.nnz if is_directed else adjacency_matrix.nnz // 2

        # Create NetworkData
        network_data = NetworkData(
            adjacency_matrix=adjacency_matrix,
            num_nodes=self.num_nodes,
            num_edges=num_edges,
            network_type=self.network_type,
            node_positions=node_positions,
            is_directed=is_directed,
            backend_type=backend_type,
            backend_graph=backend_graph,
        )

        return network_data

    def visualize_network(
        self,
        node_values: Optional[np.ndarray] = None,
        edge_values: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Any:
        """Visualize network structure with optional node/edge values."""
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX required for network visualization")

        # Implementation will be in visualization module
        pass


class GridNetwork(BaseNetworkGeometry):
    """Grid/lattice network for MFG problems."""

    def __init__(
        self,
        width: int,
        height: int = None,
        periodic: bool = False,
        backend_preference: NetworkBackendType = NetworkBackendType.IGRAPH,
    ):
        self.width = width
        self.height = height or width
        self.periodic = periodic
        super().__init__(self.width * self.height, NetworkType.GRID, backend_preference)

    def create_network(self, **kwargs) -> NetworkData:
        """Create grid network using optimal backend."""
        nodes = self.width * self.height

        # Choose optimal backend for this size
        backend_type = self.backend_manager.choose_backend(
            nodes, OperationType.GENERAL, force_backend=kwargs.get("force_backend")
        )
        backend = self.backend_manager.get_backend(backend_type)

        # Create graph using chosen backend
        graph = backend.create_graph(nodes, directed=False)

        # Generate node positions
        positions = np.zeros((nodes, 2))
        edges = []

        for i in range(self.height):
            for j in range(self.width):
                node_id = i * self.width + j
                positions[node_id] = [j, i]

                # Connect to neighbors
                neighbors = self._get_grid_neighbors(i, j)
                for ni, nj in neighbors:
                    neighbor_id = ni * self.width + nj
                    if node_id < neighbor_id:  # Avoid duplicate edges
                        edges.append((node_id, neighbor_id))

        # Add edges to backend graph
        if edges:
            backend.add_edges(graph, edges)

        # Convert to NetworkData
        self.network_data = self._create_network_data_from_backend(
            graph, backend_type, positions
        )

        return self.network_data

    def _get_grid_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """Get grid neighbors with optional periodic boundary conditions."""
        neighbors = []

        # 4-connected grid (can be extended to 8-connected)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for di, dj in directions:
            ni, nj = i + di, j + dj

            if self.periodic:
                ni = ni % self.height
                nj = nj % self.width
                neighbors.append((ni, nj))
            else:
                if 0 <= ni < self.height and 0 <= nj < self.width:
                    neighbors.append((ni, nj))

        return neighbors

    def compute_distance_matrix(self) -> np.ndarray:
        """Compute Manhattan distances on grid."""
        distances = np.zeros((self.num_nodes, self.num_nodes))

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                i1, j1 = divmod(i, self.width)
                i2, j2 = divmod(j, self.width)
                distances[i, j] = abs(i1 - i2) + abs(j1 - j2)

        return distances


class RandomNetwork(BaseNetworkGeometry):
    """Random network (Erdős–Rényi model) for MFG problems."""

    def __init__(
        self,
        num_nodes: int,
        connection_prob: float = 0.1,
        backend_preference: NetworkBackendType = NetworkBackendType.IGRAPH,
    ):
        self.connection_prob = connection_prob
        super().__init__(num_nodes, NetworkType.RANDOM, backend_preference)

    def create_network(self, seed: Optional[int] = None, **kwargs) -> NetworkData:
        """Create random network using optimal backend."""
        if seed is not None:
            np.random.seed(seed)

        # Choose optimal backend for this size
        backend_type = self.backend_manager.choose_backend(
            self.num_nodes,
            OperationType.GENERAL,
            force_backend=kwargs.get("force_backend"),
        )
        backend = self.backend_manager.get_backend(backend_type)

        # Create graph using chosen backend
        graph = backend.create_graph(self.num_nodes, directed=False)

        # Generate random edges
        edges = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if np.random.random() < self.connection_prob:
                    edges.append((i, j))

        # Add edges to backend graph
        if edges:
            backend.add_edges(graph, edges)

        # Generate random node positions for visualization
        positions = np.random.rand(self.num_nodes, 2)

        # Convert to NetworkData
        self.network_data = self._create_network_data_from_backend(
            graph, backend_type, positions
        )

        return self.network_data

    def compute_distance_matrix(self) -> np.ndarray:
        """Compute shortest path distances using BFS."""
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX required for shortest path computation")

        G = nx.from_scipy_sparse_array(self.network_data.adjacency_matrix)
        return np.array(nx.floyd_warshall_matrix(G))


class ScaleFreeNetwork(BaseNetworkGeometry):
    """Scale-free network (Barabási–Albert model) for MFG problems."""

    def __init__(
        self,
        num_nodes: int,
        num_edges_per_node: int = 2,
        backend_preference: NetworkBackendType = NetworkBackendType.IGRAPH,
    ):
        self.num_edges_per_node = num_edges_per_node
        super().__init__(num_nodes, NetworkType.SCALE_FREE, backend_preference)

    def create_network(self, seed: Optional[int] = None, **kwargs) -> NetworkData:
        """Create scale-free network using optimal backend."""
        if seed is not None:
            np.random.seed(seed)

        # Choose optimal backend for this size
        backend_type = self.backend_manager.choose_backend(
            self.num_nodes,
            OperationType.GENERAL,
            force_backend=kwargs.get("force_backend"),
        )
        backend = self.backend_manager.get_backend(backend_type)

        # Create graph using chosen backend
        graph = backend.create_graph(self.num_nodes, directed=False)

        # Generate Barabási-Albert network edges
        edges = self._generate_barabasi_albert_edges(seed)

        # Add edges to backend graph
        if edges:
            backend.add_edges(graph, edges)

        # Generate positions using spring-like layout
        positions = self._generate_spring_positions(seed)

        # Convert to NetworkData
        self.network_data = self._create_network_data_from_backend(
            graph, backend_type, positions
        )

        return self.network_data

    def _generate_barabasi_albert_edges(
        self, seed: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """Generate Barabási-Albert network edges using preferential attachment."""
        if seed is not None:
            np.random.seed(seed)

        edges = []
        node_degrees = np.zeros(self.num_nodes)

        # Start with a complete graph of m+1 nodes
        initial_nodes = min(self.num_edges_per_node + 1, self.num_nodes)
        for i in range(initial_nodes):
            for j in range(i + 1, initial_nodes):
                edges.append((i, j))
                node_degrees[i] += 1
                node_degrees[j] += 1

        # Add remaining nodes with preferential attachment
        for new_node in range(initial_nodes, self.num_nodes):
            # Select m existing nodes to connect to based on their degrees
            targets = []
            total_degree = np.sum(node_degrees[:new_node])

            if total_degree > 0:
                # Preferential attachment: probability proportional to degree
                probabilities = node_degrees[:new_node] / total_degree
                targets = np.random.choice(
                    new_node,
                    size=min(self.num_edges_per_node, new_node),
                    replace=False,
                    p=probabilities,
                )
            else:
                # Fallback: random attachment
                targets = np.random.choice(
                    new_node, size=min(self.num_edges_per_node, new_node), replace=False
                )

            # Add edges
            for target in targets:
                edges.append((new_node, target))
                node_degrees[new_node] += 1
                node_degrees[target] += 1

        return edges

    def _generate_spring_positions(self, seed: Optional[int] = None) -> np.ndarray:
        """Generate node positions using spring-like layout."""
        if seed is not None:
            np.random.seed(seed)

        # Simple circular layout as fallback
        angles = np.linspace(0, 2 * np.pi, self.num_nodes, endpoint=False)
        radius = 1.0
        positions = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

        # Add some randomness
        positions += 0.1 * np.random.randn(self.num_nodes, 2)

        return positions

    def compute_distance_matrix(self) -> np.ndarray:
        """Compute shortest path distances."""
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX required for shortest path computation")

        G = nx.from_scipy_sparse_array(self.network_data.adjacency_matrix)
        return np.array(nx.floyd_warshall_matrix(G))


# Factory function for creating networks
def create_network(
    network_type: Union[str, NetworkType],
    num_nodes: int,
    backend_preference: NetworkBackendType = NetworkBackendType.IGRAPH,
    **kwargs,
) -> BaseNetworkGeometry:
    """
    Factory function for creating network geometries with backend selection.

    Args:
        network_type: Type of network to create
        num_nodes: Number of nodes in network
        backend_preference: Preferred backend (igraph, networkit, networkx)
        **kwargs: Network-specific parameters

    Returns:
        Configured network geometry object with optimal backend
    """
    if isinstance(network_type, str):
        network_type = NetworkType(network_type)

    if network_type == NetworkType.GRID:
        width = kwargs.get("width", int(np.sqrt(num_nodes)))
        height = kwargs.get("height", num_nodes // width)
        periodic = kwargs.get("periodic", False)
        return GridNetwork(width, height, periodic, backend_preference)

    elif network_type == NetworkType.RANDOM:
        connection_prob = kwargs.get("connection_prob", 0.1)
        return RandomNetwork(num_nodes, connection_prob, backend_preference)

    elif network_type == NetworkType.SCALE_FREE:
        num_edges_per_node = kwargs.get("num_edges_per_node", 2)
        return ScaleFreeNetwork(num_nodes, num_edges_per_node, backend_preference)

    else:
        raise ValueError(f"Unsupported network type: {network_type}")


# Utility functions for network analysis
def compute_network_statistics(network_data: NetworkData) -> Dict[str, float]:
    """Compute comprehensive network statistics."""
    A = network_data.adjacency_matrix
    stats = {
        "num_nodes": network_data.num_nodes,
        "num_edges": network_data.num_edges,
        "density": network_data.metadata.get("density", 0),
        "average_degree": network_data.metadata.get("average_degree", 0),
        "max_degree": network_data.metadata.get("max_degree", 0),
        "clustering_coefficient": network_data.metadata.get(
            "clustering_coefficient", 0
        ),
        "is_connected": network_data.is_connected,
        "is_directed": network_data.is_directed,
        "is_weighted": network_data.is_weighted,
    }

    # Add spectral properties if possible
    try:
        L = network_data.laplacian_matrix.toarray()
        eigenvals = np.linalg.eigvals(L)
        eigenvals = np.sort(eigenvals)

        stats.update(
            {
                "algebraic_connectivity": eigenvals[1] if len(eigenvals) > 1 else 0,
                "spectral_gap": (
                    eigenvals[1] - eigenvals[0] if len(eigenvals) > 1 else 0
                ),
                "largest_eigenvalue": eigenvals[-1],
            }
        )
    except Exception:
        # Skip spectral analysis if it fails
        pass

    return stats
