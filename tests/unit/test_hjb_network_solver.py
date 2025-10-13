#!/usr/bin/env python3
"""
Unit tests for NetworkHJBSolver.

Tests the HJB solver for Mean Field Games on network/graph structures,
including various time discretization schemes and network operators.
"""

import pytest

import numpy as np

from mfg_pde.alg.numerical.hjb_solvers.hjb_network import NetworkHJBSolver
from mfg_pde.core.network_mfg_problem import NetworkMFGProblem
from mfg_pde.geometry.network_geometry import GridNetwork

# Skip all tests if igraph is not available (network backend dependency)
igraph = pytest.importorskip("igraph")


class TestNetworkHJBSolverInitialization:
    """Test NetworkHJBSolver initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic solver initialization with default parameters."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem)

        assert solver.hjb_method_name == "NetworkHJB_explicit"
        assert solver.scheme == "explicit"
        assert solver.cfl_factor == 0.5
        assert solver.max_iterations == 1000
        assert solver.tolerance == 1e-6

    def test_explicit_scheme_initialization(self):
        """Test initialization with explicit scheme."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="explicit")

        assert solver.scheme == "explicit"
        assert solver.hjb_method_name == "NetworkHJB_explicit"

    def test_implicit_scheme_initialization(self):
        """Test initialization with implicit scheme."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="implicit")

        assert solver.scheme == "implicit"
        assert solver.hjb_method_name == "NetworkHJB_implicit"

    def test_semi_implicit_scheme_initialization(self):
        """Test initialization with semi-implicit scheme."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="semi_implicit")

        assert solver.scheme == "semi_implicit"
        assert solver.hjb_method_name == "NetworkHJB_semi_implicit"

    def test_custom_cfl_factor(self):
        """Test initialization with custom CFL factor."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="explicit", cfl_factor=0.3)

        assert solver.cfl_factor == 0.3

    def test_custom_iteration_parameters(self):
        """Test initialization with custom iteration parameters."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = NetworkHJBSolver(
            problem,
            scheme="implicit",
            max_iterations=500,
            tolerance=1e-8,
        )

        assert solver.max_iterations == 500
        assert solver.tolerance == 1e-8

    def test_network_properties_extracted(self):
        """Test that network properties are properly extracted."""
        network = GridNetwork(width=4, height=4)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem)

        assert solver.num_nodes == 16
        assert solver.adjacency_matrix is not None
        assert solver.laplacian_matrix is not None

    def test_time_discretization(self):
        """Test that time discretization is properly computed."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=2.0,
            Nt=20,
        )

        solver = NetworkHJBSolver(problem)

        assert np.isclose(solver.dt, 0.1)
        assert len(solver.times) == 21
        assert np.isclose(solver.times[0], 0.0)
        assert np.isclose(solver.times[-1], 2.0)

    def test_gradient_operators_initialized(self):
        """Test that gradient operators are initialized."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem)

        assert hasattr(solver, "gradient_ops")
        assert len(solver.gradient_ops) == solver.num_nodes


class TestNetworkHJBSolverSolveHJBSystem:
    """Test the main solve_hjb_system method."""

    def test_solve_hjb_system_shape(self):
        """Test that solve_hjb_system returns correct shape."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="implicit")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        # Create inputs
        M_density = np.ones((Nt, num_nodes))
        U_final = np.zeros(num_nodes)

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final)

        assert U_solution.shape == (Nt, num_nodes)
        assert np.all(np.isfinite(U_solution))

    def test_solve_hjb_system_final_condition(self):
        """Test that final condition is preserved."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="implicit")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        # Create inputs with specific final condition
        M_density = np.ones((Nt, num_nodes))
        U_final = np.random.rand(num_nodes)

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final)

        # Final time step should match final condition
        assert np.allclose(U_solution[-1, :], U_final)

    def test_solve_with_explicit_scheme(self):
        """Test solving with explicit scheme."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.1,  # Short time for stability
            Nt=20,
        )

        solver = NetworkHJBSolver(problem, scheme="explicit", cfl_factor=0.3)

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        M_density = np.ones((Nt, num_nodes))
        U_final = np.zeros(num_nodes)

        U_solution = solver.solve_hjb_system(M_density, U_final)

        assert np.all(np.isfinite(U_solution))

    def test_solve_with_implicit_scheme(self):
        """Test solving with implicit scheme."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="implicit")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        M_density = np.ones((Nt, num_nodes))
        U_final = np.zeros(num_nodes)

        U_solution = solver.solve_hjb_system(M_density, U_final)

        assert np.all(np.isfinite(U_solution))

    def test_solve_with_semi_implicit_scheme(self):
        """Test solving with semi-implicit scheme."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="semi_implicit")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        M_density = np.ones((Nt, num_nodes))
        U_final = np.zeros(num_nodes)

        U_solution = solver.solve_hjb_system(M_density, U_final)

        assert np.all(np.isfinite(U_solution))

    def test_solve_with_varying_density(self):
        """Test solving with non-uniform density."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="implicit")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        # Create non-uniform density
        M_density = np.random.rand(Nt, num_nodes)
        M_density = M_density / np.sum(M_density, axis=1, keepdims=True)

        U_final = np.zeros(num_nodes)

        U_solution = solver.solve_hjb_system(M_density, U_final)

        assert np.all(np.isfinite(U_solution))

    def test_invalid_scheme_raises_error(self):
        """Test that invalid scheme raises error."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="explicit")
        solver.scheme = "invalid_scheme"  # Manually set invalid scheme

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        M_density = np.ones((Nt, num_nodes))
        U_final = np.zeros(num_nodes)

        with pytest.raises(ValueError, match="Unknown scheme"):
            solver.solve_hjb_system(M_density, U_final)


class TestNetworkHJBSolverNumericalProperties:
    """Test numerical properties of network HJB solutions."""

    def test_solution_finiteness(self):
        """Test that solution remains finite throughout."""
        network = GridNetwork(width=4, height=4)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=15,
        )

        solver = NetworkHJBSolver(problem, scheme="implicit")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        M_density = np.ones((Nt, num_nodes)) * 0.5
        U_final = np.random.rand(num_nodes)

        U_solution = solver.solve_hjb_system(M_density, U_final)

        # All values should be finite
        assert np.all(np.isfinite(U_solution))

    def test_backward_time_propagation(self):
        """Test that solution propagates backward in time."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="implicit")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        M_density = np.ones((Nt, num_nodes))
        U_final = np.ones(num_nodes)  # Non-zero final condition

        U_solution = solver.solve_hjb_system(M_density, U_final)

        # Solution should propagate backward (not remain zero at t=0)
        assert not np.allclose(U_solution[0, :], 0.0)


class TestNetworkHJBSolverDifferentNetworks:
    """Test solver with different network geometries."""

    def test_small_grid_network(self):
        """Test solver on small grid network."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="implicit")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        M_density = np.ones((Nt, num_nodes))
        U_final = np.zeros(num_nodes)

        U_solution = solver.solve_hjb_system(M_density, U_final)

        assert U_solution.shape == (Nt, num_nodes)
        assert np.all(np.isfinite(U_solution))

    def test_rectangular_grid_network(self):
        """Test solver on non-square grid."""
        network = GridNetwork(width=4, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="implicit")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        M_density = np.ones((Nt, num_nodes))
        U_final = np.zeros(num_nodes)

        U_solution = solver.solve_hjb_system(M_density, U_final)

        assert num_nodes == 12
        assert U_solution.shape == (Nt, 12)

    def test_periodic_grid_network(self):
        """Test solver on periodic grid."""
        network = GridNetwork(width=4, height=4, periodic=True)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="implicit")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        M_density = np.ones((Nt, num_nodes))
        U_final = np.zeros(num_nodes)

        U_solution = solver.solve_hjb_system(M_density, U_final)

        assert np.all(np.isfinite(U_solution))


class TestNetworkHJBSolverIntegration:
    """Integration tests with actual MFG problems."""

    def test_solver_not_abstract(self):
        """Test that NetworkHJBSolver can be instantiated."""
        import inspect

        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        # Should not raise TypeError about abstract methods
        solver = NetworkHJBSolver(problem)
        assert isinstance(solver, NetworkHJBSolver)

        # Should not have abstract methods
        assert not inspect.isabstract(NetworkHJBSolver)

    def test_solver_with_different_parameters(self):
        """Test solver with various parameter configurations."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        configs = [
            {"scheme": "explicit", "cfl_factor": 0.4},
            {"scheme": "implicit", "max_iterations": 500},
            {"scheme": "semi_implicit", "tolerance": 1e-7},
        ]

        for config in configs:
            problem = NetworkMFGProblem(
                network_geometry=network,
                T=0.2,
                Nt=10,
            )

            solver = NetworkHJBSolver(problem, **config)

            Nt = problem.Nt + 1
            num_nodes = problem.num_nodes

            M_density = np.ones((Nt, num_nodes))
            U_final = np.zeros(num_nodes)

            U_solution = solver.solve_hjb_system(M_density, U_final)

            assert np.all(np.isfinite(U_solution))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
