"""
Network Mean Field Games Problem Formulation.

This module implements MFG problems on network/graph structures, extending
the continuous MFG framework to discrete network domains.

Mathematical Framework:
- State space: Discrete nodes of the network
- Density evolution: Network flow dynamics on edges
- HJB equation: Discrete optimal control on graphs
- Coupling: Local interactions at nodes and along edges

Key differences from continuous MFG:
- Spatial derivatives → Graph discrete derivatives (differences)
- Laplacian operator → Graph Laplacian matrix
- Continuous density → Discrete node masses/flows
- Boundary conditions → Network boundary nodes
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from ..geometry.network_geometry import BaseNetworkGeometry, NetworkData
from .mfg_problem import MFGProblem


@dataclass
class NetworkMFGComponents:
    """
    Components for defining MFG problems on networks.

    This extends the continuous MFGComponents to handle discrete network structures,
    including support for Lagrangian formulations and trajectory measures.
    """

    # Network-specific Hamiltonian (depends on node states and edge flows)
    hamiltonian_func: Optional[Callable] = None  # H(node, neighbors, m, p, t)
    hamiltonian_dm_func: Optional[Callable] = None  # dH/dm at nodes

    # Lagrangian formulation support (based on ArXiv 2207.10908v3)
    lagrangian_func: Optional[Callable] = None  # L(node, velocity, m, t)
    velocity_space_dim: int = 2  # Dimension of velocity space
    trajectory_cost_func: Optional[Callable] = None  # Cost along trajectories
    relaxed_control: bool = False  # Use relaxed equilibria

    # Node-based potential function
    node_potential_func: Optional[Callable] = None  # V(node, t)

    # Edge-based costs/rewards
    edge_cost_func: Optional[Callable] = None  # Cost of moving along edges
    congestion_func: Optional[Callable] = None  # Congestion effects

    # Initial and terminal conditions on network
    initial_node_density_func: Optional[Callable] = None  # m_0(node)
    terminal_node_value_func: Optional[Callable] = None  # u_T(node)

    # Network boundary conditions
    boundary_nodes: Optional[List[int]] = None  # Nodes with boundary conditions
    boundary_values_func: Optional[Callable] = None  # Boundary values

    # Flow dynamics parameters
    diffusion_coefficient: float = 1.0  # Diffusion strength
    drift_coefficient: float = 1.0  # Drift/advection strength

    # Network-specific coupling
    node_interaction_func: Optional[Callable] = None  # Local node interactions
    edge_interaction_func: Optional[Callable] = None  # Edge-based interactions

    # Problem parameters
    problem_params: Dict[str, Any] = field(default_factory=dict)


class NetworkMFGProblem(MFGProblem):
    """
    Mean Field Games problem on network structures.

    This class implements MFG formulations on discrete network domains,
    supporting various network topologies and interaction mechanisms.

    Mathematical formulation:

    HJB equation (discrete):
    ∂u/∂t + H_i(m, ∇_G u, t) = 0  at node i
    u(T, i) = g(i)                  terminal condition

    Fokker-Planck equation (discrete):
    ∂m/∂t - div_G(m ∇_G H_p) - Δ_G m = 0  on network
    m(0, i) = m_0(i)                        initial condition

    where:
    - ∇_G: Graph gradient operator
    - div_G: Graph divergence operator
    - Δ_G: Graph Laplacian operator
    - H_i: Hamiltonian at node i
    """

    def __init__(
        self,
        network_geometry: BaseNetworkGeometry,
        T: float = 1.0,
        Nt: int = 100,
        components: Optional[NetworkMFGComponents] = None,
        problem_name: str = "NetworkMFG",
    ):
        """
        Initialize network MFG problem.

        Args:
            network_geometry: Network structure and geometry
            T: Terminal time
            Nt: Number of time steps
            components: Network MFG components (optional)
            problem_name: Problem identifier
        """
        # Network properties - set first before calling super()
        self.network_geometry = network_geometry
        self.network_data = network_geometry.network_data

        # Override spatial properties for network (set as private to avoid property conflicts)
        self._xmin = 0.0
        self._xmax = float(network_geometry.num_nodes - 1)
        self._Nx = network_geometry.num_nodes - 1

        # Initialize with dummy spatial parameters (networks are discrete)
        super().__init__(T=T, Nt=Nt, xmin=self._xmin, xmax=self._xmax, Nx=self._Nx)

        self.components = components or NetworkMFGComponents()
        self.problem_name = problem_name

        # Override spatial properties for network
        self.is_network_problem = True
        self.num_nodes = network_geometry.num_nodes
        self.spatial_dimension = 0  # Discrete network, not continuous space

        # Network-specific matrices
        self.adjacency_matrix: Optional[csr_matrix] = None
        self.laplacian_matrix: Optional[csr_matrix] = None
        self.incidence_matrix: Optional[csr_matrix] = None

        self._initialize_network_operators()

    def _initialize_network_operators(self):
        """Initialize network-specific operators and matrices."""
        if self.network_data is None:
            raise ValueError("Network data not available. Create network first.")

        self.adjacency_matrix = self.network_data.adjacency_matrix
        self.laplacian_matrix = self.network_data.laplacian_matrix
        self.incidence_matrix = self.network_data.incidence_matrix

        # Store network properties
        self.is_directed = self.network_data.is_directed
        self.is_weighted = self.network_data.is_weighted
        self.num_edges = self.network_data.num_edges

    # Network-specific MFG components

    def hamiltonian(
        self, node: int, neighbors: List[int], m: np.ndarray, p: np.ndarray, t: float
    ) -> float:
        """
        Network Hamiltonian function.

        Args:
            node: Current node index
            neighbors: List of neighboring nodes
            m: Density vector at all nodes
            p: Co-state vector at all nodes
            t: Current time

        Returns:
            Hamiltonian value at the node
        """
        if self.components.hamiltonian_func is not None:
            return self.components.hamiltonian_func(node, neighbors, m, p, t)

        # Default quadratic Hamiltonian with network structure
        return self._default_network_hamiltonian(node, neighbors, m, p, t)

    def _default_network_hamiltonian(
        self, node: int, neighbors: List[int], m: np.ndarray, p: np.ndarray, t: float
    ) -> float:
        """Default network Hamiltonian implementation."""
        # Quadratic control cost + potential + density coupling
        control_cost = 0.0

        # Sum over possible moves to neighbors
        for neighbor in neighbors:
            edge_weight = self.network_data.get_edge_weight(node, neighbor)
            # Control cost for moving to neighbor
            dp = p[neighbor] - p[node]  # Discrete gradient
            control_cost += 0.5 * edge_weight * dp**2

        # Node potential
        potential = self.node_potential(node, t)

        # Density coupling (congestion effects)
        coupling = self.density_coupling(node, m, t)

        return control_cost + potential + coupling

    def hamiltonian_dm(
        self, node: int, neighbors: List[int], m: np.ndarray, p: np.ndarray, t: float
    ) -> float:
        """Derivative of Hamiltonian with respect to density."""
        if self.components.hamiltonian_dm_func is not None:
            return self.components.hamiltonian_dm_func(node, neighbors, m, p, t)

        # Default: derivative of density coupling term
        return self._default_density_coupling_derivative(node, m, t)

    # Lagrangian formulation methods (based on ArXiv 2207.10908v3)

    def lagrangian(
        self, node: int, velocity: np.ndarray, m: np.ndarray, t: float
    ) -> float:
        """
        Lagrangian function for network MFG.

        Based on the Lagrangian formulation from ArXiv 2207.10908v3,
        this represents the cost of being at a node with given velocity.

        Args:
            node: Current node index
            velocity: Velocity vector in network space
            m: Density distribution over network
            t: Current time

        Returns:
            Lagrangian value L(node, velocity, m, t)
        """
        if self.components.lagrangian_func is not None:
            return self.components.lagrangian_func(node, velocity, m, t)

        # Default Lagrangian: kinetic energy + potential + interaction
        kinetic_energy = 0.5 * np.linalg.norm(velocity) ** 2
        potential = self.node_potential(node, t)
        interaction = self.density_coupling(node, m, t)

        return kinetic_energy + potential + interaction

    def trajectory_cost(
        self,
        trajectory: List[int],
        velocities: np.ndarray,
        m_evolution: np.ndarray,
        times: np.ndarray,
    ) -> float:
        """
        Compute cost along a network trajectory.

        Args:
            trajectory: Sequence of nodes visited
            velocities: Velocity at each time step
            m_evolution: Density evolution over time
            times: Time points

        Returns:
            Total trajectory cost
        """
        if self.components.trajectory_cost_func is not None:
            return self.components.trajectory_cost_func(
                trajectory, velocities, m_evolution, times
            )

        # Default: integrate Lagrangian along trajectory
        total_cost = 0.0
        dt = times[1] - times[0] if len(times) > 1 else 1.0

        for i, (node, t) in enumerate(zip(trajectory, times)):
            if i < len(velocities):
                velocity = (
                    velocities[i] if velocities.ndim > 1 else np.array([velocities[i]])
                )
                m_current = m_evolution[i] if m_evolution.ndim > 1 else m_evolution
                lagrangian_value = self.lagrangian(node, velocity, m_current, t)
                total_cost += lagrangian_value * dt

        return total_cost

    def compute_relaxed_equilibrium(
        self, trajectory_measures: List[Callable]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute relaxed equilibrium as probability measures on trajectories.

        Based on the relaxed equilibria concept from ArXiv 2207.10908v3.

        Args:
            trajectory_measures: List of probability measures on trajectory space

        Returns:
            (u, m) where u is value function and m is density
        """
        # This is a placeholder for advanced trajectory measure computation
        # Full implementation would require sophisticated measure theory

        u = np.zeros((self.Nt + 1, self.num_nodes))
        m = np.zeros((self.Nt + 1, self.num_nodes))

        # Initialize with uniform distribution
        m[0, :] = 1.0 / self.num_nodes

        # Simple trajectory-based computation (to be enhanced)
        for t_idx in range(self.Nt):
            # Update based on trajectory measures
            for node in range(self.num_nodes):
                # Aggregate trajectory contributions
                total_measure = 0.0
                for measure in trajectory_measures:
                    total_measure += measure(node, t_idx)
                m[t_idx + 1, node] = total_measure

        # Normalize density
        for t_idx in range(self.Nt + 1):
            total = np.sum(m[t_idx, :])
            if total > 1e-12:
                m[t_idx, :] /= total

        return u, m

    def node_potential(self, node: int, t: float) -> float:
        """Potential function at network nodes."""
        if self.components.node_potential_func is not None:
            return self.components.node_potential_func(node, t)
        return 0.0

    def density_coupling(self, node: int, m: np.ndarray, t: float) -> float:
        """Density coupling/interaction at nodes."""
        if self.components.node_interaction_func is not None:
            return self.components.node_interaction_func(node, m, t)

        # Default: quadratic congestion at nodes
        return 0.5 * m[node] ** 2

    def _default_density_coupling_derivative(
        self, node: int, m: np.ndarray, t: float
    ) -> float:
        """Derivative of default density coupling."""
        return m[node]  # d/dm[i] (0.5 * m[i]^2) = m[i]

    def edge_cost(self, node_from: int, node_to: int, t: float) -> float:
        """Cost of moving along network edges."""
        if self.components.edge_cost_func is not None:
            return self.components.edge_cost_func(node_from, node_to, t)

        # Default: unit cost weighted by edge weight
        edge_weight = self.network_data.get_edge_weight(node_from, node_to)
        return edge_weight

    # Initial and terminal conditions

    def get_initial_density(self) -> np.ndarray:
        """Initial density distribution on network nodes."""
        if self.components.initial_node_density_func is not None:
            return np.array(
                [
                    self.components.initial_node_density_func(i)
                    for i in range(self.num_nodes)
                ]
            )

        # Default: uniform distribution
        initial_density = np.ones(self.num_nodes) / self.num_nodes
        return initial_density

    def get_terminal_value(self) -> np.ndarray:
        """Terminal value function on network nodes."""
        if self.components.terminal_node_value_func is not None:
            return np.array(
                [
                    self.components.terminal_node_value_func(i)
                    for i in range(self.num_nodes)
                ]
            )

        # Default: zero terminal values
        return np.zeros(self.num_nodes)

    # Network-specific operators

    def compute_graph_gradient(self, u: np.ndarray) -> np.ndarray:
        """
        Compute discrete gradient on network.

        For each node, computes differences to neighboring nodes.

        Args:
            u: Node values

        Returns:
            Gradient information (implementation dependent)
        """
        gradients = np.zeros((self.num_nodes, self.num_nodes))

        for i in range(self.num_nodes):
            neighbors = self.network_data.get_neighbors(i)
            for j in neighbors:
                gradients[i, j] = u[j] - u[i]

        return gradients

    def compute_graph_divergence(self, flow: np.ndarray) -> np.ndarray:
        """
        Compute discrete divergence on network.

        Args:
            flow: Edge flow values

        Returns:
            Divergence at each node
        """
        # Simplified: divergence as sum of incoming/outgoing flows
        divergence = np.zeros(self.num_nodes)

        # This is a simplified implementation
        # Full implementation would handle edge-based flows properly
        for i in range(self.num_nodes):
            neighbors = self.network_data.get_neighbors(i)
            div_i = 0.0
            for j in neighbors:
                # Flow from j to i minus flow from i to j
                div_i += flow[j] - flow[i]  # Simplified
            divergence[i] = div_i

        return divergence

    def apply_graph_laplacian(
        self, u: np.ndarray, coefficient: float = 1.0
    ) -> np.ndarray:
        """
        Apply graph Laplacian operator to node values.

        Args:
            u: Node values
            coefficient: Diffusion coefficient

        Returns:
            Laplacian applied to u
        """
        return coefficient * (self.laplacian_matrix @ u)

    # Boundary conditions for networks

    def apply_boundary_conditions(self, u: np.ndarray, t: float) -> np.ndarray:
        """Apply boundary conditions to network nodes."""
        u_bc = u.copy()

        if self.components.boundary_nodes is not None:
            for node in self.components.boundary_nodes:
                if self.components.boundary_values_func is not None:
                    u_bc[node] = self.components.boundary_values_func(node, t)
                else:
                    # Default: zero boundary values
                    u_bc[node] = 0.0

        return u_bc

    # Legacy interface compatibility

    def get_initial_m(self) -> np.ndarray:
        """Get initial density (legacy interface)."""
        return self.get_initial_density()

    def get_final_u(self) -> np.ndarray:
        """Get terminal value function (legacy interface)."""
        return self.get_terminal_value()

    @property
    def Nx(self) -> int:
        """Number of spatial points (nodes in network)."""
        return self._Nx

    @Nx.setter
    def Nx(self, value: int):
        """Set number of spatial points."""
        self._Nx = value

    @property
    def xmin(self) -> float:
        """Minimum spatial coordinate (dummy for networks)."""
        return self._xmin

    @xmin.setter
    def xmin(self, value: float):
        """Set minimum spatial coordinate."""
        self._xmin = value

    @property
    def xmax(self) -> float:
        """Maximum spatial coordinate (dummy for networks)."""
        return self._xmax

    @xmax.setter
    def xmax(self, value: float):
        """Set maximum spatial coordinate."""
        self._xmax = value

    @property
    def Dx(self) -> float:
        """Spatial step size (dummy for networks)."""
        return getattr(self, "_Dx", 1.0)

    @Dx.setter
    def Dx(self, value: float):
        """Set spatial step size."""
        self._Dx = value

    # Network-specific properties

    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics."""
        from ..geometry.network_geometry import compute_network_statistics

        return compute_network_statistics(self.network_data)

    def get_adjacency_matrix(self) -> csr_matrix:
        """Get network adjacency matrix."""
        return self.adjacency_matrix

    def get_laplacian_matrix(self) -> csr_matrix:
        """Get network Laplacian matrix."""
        return self.laplacian_matrix

    def get_node_neighbors(self, node: int) -> List[int]:
        """Get neighbors of a specific node."""
        return self.network_data.get_neighbors(node)

    def __str__(self) -> str:
        """String representation of network MFG problem."""
        stats = self.get_network_statistics()
        return (
            f"NetworkMFGProblem({self.problem_name})\n"
            f"  Network: {self.network_data.network_type.value}\n"
            f"  Nodes: {self.num_nodes}, Edges: {self.num_edges}\n"
            f"  Time: T={self.T}, Nt={self.Nt}\n"
            f"  Connected: {stats['is_connected']}\n"
            f"  Average degree: {stats['average_degree']:.2f}"
        )


# Factory functions for common network MFG problems


def create_grid_mfg_problem(
    width: int,
    height: int = None,
    T: float = 1.0,
    Nt: int = 100,
    periodic: bool = False,
    **kwargs,
) -> NetworkMFGProblem:
    """Create MFG problem on grid network."""
    from ..geometry.network_geometry import GridNetwork

    height = height or width
    network = GridNetwork(width, height, periodic)
    network.create_network()

    components = NetworkMFGComponents(**kwargs)

    return NetworkMFGProblem(
        network_geometry=network,
        T=T,
        Nt=Nt,
        components=components,
        problem_name=f"GridMFG_{width}x{height}",
    )


def create_random_mfg_problem(
    num_nodes: int,
    connection_prob: float = 0.1,
    T: float = 1.0,
    Nt: int = 100,
    seed: Optional[int] = None,
    **kwargs,
) -> NetworkMFGProblem:
    """Create MFG problem on random network."""
    from ..geometry.network_geometry import RandomNetwork

    network = RandomNetwork(num_nodes, connection_prob)
    network.create_network(seed=seed)

    components = NetworkMFGComponents(**kwargs)

    return NetworkMFGProblem(
        network_geometry=network,
        T=T,
        Nt=Nt,
        components=components,
        problem_name=f"RandomMFG_N{num_nodes}_p{connection_prob}",
    )


def create_scale_free_mfg_problem(
    num_nodes: int,
    num_edges_per_node: int = 2,
    T: float = 1.0,
    Nt: int = 100,
    seed: Optional[int] = None,
    **kwargs,
) -> NetworkMFGProblem:
    """Create MFG problem on scale-free network."""
    from ..geometry.network_geometry import ScaleFreeNetwork

    network = ScaleFreeNetwork(num_nodes, num_edges_per_node)
    network.create_network(seed=seed)

    components = NetworkMFGComponents(**kwargs)

    return NetworkMFGProblem(
        network_geometry=network,
        T=T,
        Nt=Nt,
        components=components,
        problem_name=f"ScaleFreeMFG_N{num_nodes}_m{num_edges_per_node}",
    )
