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
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, diags
from scipy.spatial.distance import pdist, squareform

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
    edge_weights: Optional[np.ndarray] = None    # Edge weights
    node_weights: Optional[np.ndarray] = None    # Node weights/capacities
    
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
    
    def __post_init__(self):
        """Compute derived network properties and matrices."""
        self._validate_network_data()
        self._compute_derived_matrices()
        self._compute_network_properties()
    
    def _validate_network_data(self):
        """Validate network data consistency."""
        assert self.adjacency_matrix.shape[0] == self.adjacency_matrix.shape[1], \
            "Adjacency matrix must be square"
        assert self.adjacency_matrix.shape[0] == self.num_nodes, \
            f"Adjacency matrix size {self.adjacency_matrix.shape[0]} != num_nodes {self.num_nodes}"
        
        if self.node_positions is not None:
            assert self.node_positions.shape[0] == self.num_nodes, \
                "Node positions must match number of nodes"
            
        if self.edge_weights is not None:
            assert len(self.edge_weights) == self.num_edges, \
                "Edge weights must match number of edges"
                
        if self.node_weights is not None:
            assert len(self.node_weights) == self.num_nodes, \
                "Node weights must match number of nodes"
    
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
        self.degree_matrix = diags(degrees, format='csr')
        
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
        self.metadata.update({
            'average_degree': np.mean(np.array(A.sum(axis=1)).flatten()),
            'max_degree': np.max(np.array(A.sum(axis=1)).flatten()),
            'density': self.num_edges / (self.num_nodes * (self.num_nodes - 1) / 2),
            'clustering_coefficient': self._compute_clustering_coefficient(),
        })
    
    def _check_connectivity(self) -> bool:
        """Check if network is connected (simplified version)."""
        if not NETWORKX_AVAILABLE:
            return True  # Assume connected if NetworkX not available
            
        # Convert to NetworkX for connectivity check
        G = nx.from_scipy_sparse_array(self.adjacency_matrix)
        return nx.is_connected(G) if not self.is_directed else nx.is_strongly_connected(G)
    
    def _compute_clustering_coefficient(self) -> float:
        """Compute average clustering coefficient."""
        if not NETWORKX_AVAILABLE:
            return 0.0
            
        G = nx.from_scipy_sparse_array(self.adjacency_matrix)
        return nx.average_clustering(G)
    
    def get_neighbors(self, node_id: int) -> List[int]:
        """Get neighbors of a given node."""
        return self.adjacency_matrix[node_id].nonzero()[1].tolist()
    
    def get_edge_weight(self, node_i: int, node_j: int) -> float:
        """Get weight of edge between two nodes."""
        return self.adjacency_matrix[node_i, node_j]
    
    def get_node_degree(self, node_id: int) -> int:
        """Get degree of a node."""
        return int(self.adjacency_matrix[node_id].sum())


class BaseNetworkGeometry(ABC):
    """
    Abstract base class for network geometries in MFG problems.
    
    This class defines the interface for creating and managing
    network structures for Mean Field Games on graphs.
    """
    
    def __init__(self, num_nodes: int, network_type: NetworkType = NetworkType.CUSTOM):
        self.num_nodes = num_nodes
        self.network_type = network_type
        self.network_data: Optional[NetworkData] = None
    
    @abstractmethod
    def create_network(self, **kwargs) -> NetworkData:
        """Create network with specified parameters."""
        pass
    
    @abstractmethod
    def compute_distance_matrix(self) -> np.ndarray:
        """Compute shortest path distances between all node pairs."""
        pass
    
    def get_laplacian_operator(self, 
                              operator_type: str = "combinatorial") -> csr_matrix:
        """
        Get graph Laplacian operator for MFG computations.
        
        Args:
            operator_type: Type of Laplacian ("combinatorial", "normalized", "random_walk")
            
        Returns:
            Sparse Laplacian matrix
        """
        if self.network_data is None:
            raise ValueError("Network data not initialized. Call create_network() first.")
            
        A = self.network_data.adjacency_matrix
        D = self.network_data.degree_matrix
        
        if operator_type == "combinatorial":
            return D - A
        elif operator_type == "normalized":
            # L_norm = D^(-1/2) * L * D^(-1/2)
            D_inv_sqrt = diags(1.0 / np.sqrt(D.diagonal() + 1e-12), format='csr')
            L = D - A
            return D_inv_sqrt @ L @ D_inv_sqrt
        elif operator_type == "random_walk":
            # L_rw = D^(-1) * L
            D_inv = diags(1.0 / (D.diagonal() + 1e-12), format='csr')
            L = D - A
            return D_inv @ L
        else:
            raise ValueError(f"Unknown Laplacian type: {operator_type}")
    
    def visualize_network(self, 
                         node_values: Optional[np.ndarray] = None,
                         edge_values: Optional[np.ndarray] = None,
                         **kwargs) -> Any:
        """Visualize network structure with optional node/edge values."""
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX required for network visualization")
            
        # Implementation will be in visualization module
        pass


class GridNetwork(BaseNetworkGeometry):
    """Grid/lattice network for MFG problems."""
    
    def __init__(self, width: int, height: int = None, periodic: bool = False):
        self.width = width
        self.height = height or width
        self.periodic = periodic
        super().__init__(self.width * self.height, NetworkType.GRID)
    
    def create_network(self, **kwargs) -> NetworkData:
        """Create grid network."""
        nodes = self.width * self.height
        
        # Create adjacency matrix for grid
        A = sp.lil_matrix((nodes, nodes))
        positions = np.zeros((nodes, 2))
        
        for i in range(self.height):
            for j in range(self.width):
                node_id = i * self.width + j
                positions[node_id] = [j, i]
                
                # Connect to neighbors
                neighbors = self._get_grid_neighbors(i, j)
                for ni, nj in neighbors:
                    neighbor_id = ni * self.width + nj
                    A[node_id, neighbor_id] = 1
        
        A = A.tocsr()
        num_edges = A.nnz // 2  # Undirected graph
        
        self.network_data = NetworkData(
            adjacency_matrix=A,
            num_nodes=nodes,
            num_edges=num_edges,
            network_type=self.network_type,
            node_positions=positions,
            is_directed=False
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
    
    def __init__(self, num_nodes: int, connection_prob: float = 0.1):
        self.connection_prob = connection_prob
        super().__init__(num_nodes, NetworkType.RANDOM)
    
    def create_network(self, seed: Optional[int] = None, **kwargs) -> NetworkData:
        """Create random network."""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random adjacency matrix
        A = sp.random(self.num_nodes, self.num_nodes, 
                     density=self.connection_prob, format='csr')
        
        # Make symmetric (undirected)
        A = A + A.T
        A.data = np.minimum(A.data, 1)  # Remove multiple edges
        A.setdiag(0)  # Remove self-loops
        
        num_edges = A.nnz // 2
        
        # Generate random node positions for visualization
        positions = np.random.rand(self.num_nodes, 2)
        
        self.network_data = NetworkData(
            adjacency_matrix=A,
            num_nodes=self.num_nodes,
            num_edges=num_edges,
            network_type=self.network_type,
            node_positions=positions,
            is_directed=False
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
    
    def __init__(self, num_nodes: int, num_edges_per_node: int = 2):
        self.num_edges_per_node = num_edges_per_node
        super().__init__(num_nodes, NetworkType.SCALE_FREE)
    
    def create_network(self, seed: Optional[int] = None, **kwargs) -> NetworkData:
        """Create scale-free network."""
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX required for scale-free network generation")
        
        # Generate Barabási–Albert network
        G = nx.barabasi_albert_graph(self.num_nodes, self.num_edges_per_node, seed=seed)
        
        # Convert to sparse matrix
        A = nx.adjacency_matrix(G, format='csr')
        num_edges = G.number_of_edges()
        
        # Generate positions using spring layout
        pos = nx.spring_layout(G, seed=seed)
        positions = np.array([[pos[i][0], pos[i][1]] for i in range(self.num_nodes)])
        
        self.network_data = NetworkData(
            adjacency_matrix=A,
            num_nodes=self.num_nodes,
            num_edges=num_edges,
            network_type=self.network_type,
            node_positions=positions,
            is_directed=False
        )
        
        return self.network_data
    
    def compute_distance_matrix(self) -> np.ndarray:
        """Compute shortest path distances."""
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX required for shortest path computation")
            
        G = nx.from_scipy_sparse_array(self.network_data.adjacency_matrix)
        return np.array(nx.floyd_warshall_matrix(G))


# Factory function for creating networks
def create_network(network_type: Union[str, NetworkType], 
                  num_nodes: int, 
                  **kwargs) -> BaseNetworkGeometry:
    """
    Factory function for creating network geometries.
    
    Args:
        network_type: Type of network to create
        num_nodes: Number of nodes in network
        **kwargs: Network-specific parameters
        
    Returns:
        Configured network geometry object
    """
    if isinstance(network_type, str):
        network_type = NetworkType(network_type)
    
    if network_type == NetworkType.GRID:
        width = kwargs.get('width', int(np.sqrt(num_nodes)))
        height = kwargs.get('height', num_nodes // width)
        periodic = kwargs.get('periodic', False)
        return GridNetwork(width, height, periodic)
    
    elif network_type == NetworkType.RANDOM:
        connection_prob = kwargs.get('connection_prob', 0.1)
        return RandomNetwork(num_nodes, connection_prob)
    
    elif network_type == NetworkType.SCALE_FREE:
        num_edges_per_node = kwargs.get('num_edges_per_node', 2)
        return ScaleFreeNetwork(num_nodes, num_edges_per_node)
    
    else:
        raise ValueError(f"Unsupported network type: {network_type}")


# Utility functions for network analysis
def compute_network_statistics(network_data: NetworkData) -> Dict[str, float]:
    """Compute comprehensive network statistics."""
    A = network_data.adjacency_matrix
    stats = {
        'num_nodes': network_data.num_nodes,
        'num_edges': network_data.num_edges,
        'density': network_data.metadata.get('density', 0),
        'average_degree': network_data.metadata.get('average_degree', 0),
        'max_degree': network_data.metadata.get('max_degree', 0),
        'clustering_coefficient': network_data.metadata.get('clustering_coefficient', 0),
        'is_connected': network_data.is_connected,
        'is_directed': network_data.is_directed,
        'is_weighted': network_data.is_weighted,
    }
    
    # Add spectral properties if possible
    try:
        L = network_data.laplacian_matrix.toarray()
        eigenvals = np.linalg.eigvals(L)
        eigenvals = np.sort(eigenvals)
        
        stats.update({
            'algebraic_connectivity': eigenvals[1] if len(eigenvals) > 1 else 0,
            'spectral_gap': eigenvals[1] - eigenvals[0] if len(eigenvals) > 1 else 0,
            'largest_eigenvalue': eigenvals[-1],
        })
    except Exception:
        # Skip spectral analysis if it fails
        pass
    
    return stats