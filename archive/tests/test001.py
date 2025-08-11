import copy
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from time import time
from typing import Callable, Optional, Tuple

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.sparse
from matplotlib.animation import FuncAnimation
from scipy.sparse import csr_matrix, diags, lil_matrix
from scipy.sparse.linalg import spsolve

''' Struct like class for boundary conditions'''


@dataclass
class BoundaryCondition:
    """Class for boundary condition configuration
    Note the dimension of the matrix operator for the boundary conditions
    periodic: M * M
    dirichlet: M-1 * M-1
    neumann: M+1 * M+1
    robin: M+1 * M+1
    """

    type: str  # 'periodic', 'dirichlet', 'neumann', or 'robin'
    # For Dirichlet: value of u
    # For Neumann: value of du/dn
    left_value: Optional[float] = None
    right_value: Optional[float] = None
    # For Robin: value of \gamma in \alpha* u + \beta* du/dn= g

    # Additional parameters for Robin boundary conditions
    left_alpha: Optional[float] = None  # coefficient of u
    left_beta: Optional[float] = None  # coefficient of du/dn
    right_alpha: Optional[float] = None  # coefficient of u
    right_beta: Optional[float] = None  # coefficient of du/dn

    def __post_init__(self):
        """Validate boundary condition parameters"""
        if self.type == 'robin':
            if any(v is None for v in [self.left_alpha, self.left_beta, self.right_alpha, self.right_beta]):
                raise ValueError("Robin boundary conditions require alpha and beta coefficients")


'''Base class for MFG solvers implementing common functionality'''


class BaseMFGSolver(metaclass=ABCMeta):
    """Base class for MFG solvers implementing common functionality"""

    def __init__(
        self,
        N: int,  # time steps
        M: int,  # space steps
        sigma,
        T=1,
        X=1,
        tol=1e-8,
        max_iter=100,
        boundary: Optional[BoundaryCondition] = None,
        m_init: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        u_term: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        **kwargs,
    ):
        self.T = T
        self.N = N
        self.X = X
        self.M = M
        self.sigma = sigma
        self.tol = tol
        self.max_iter = max_iter
        self.debug = False
        # Extract functions from kwargs or use defaults
        self.boundary = kwargs.get('boundary', BoundaryCondition(type='periodic'))
        # Don't override m_init and u_term if they were set in parent
        self.m_init = m_init
        if self.m_init is None:
            self.m_init = kwargs.get('m_init', lambda x: 1 + 0.5 * np.cos(4 * np.pi * x))
        self.u_term = u_term
        if self.u_term is None:
            self.u_term = kwargs.get('u_term', lambda x: np.zeros_like(x))

        self.potential = kwargs.get('potential', lambda x: np.sin(2 * np.pi * x))
        self.hamiltonian = kwargs.get('hamiltonian', lambda x, p: 0.5 * p**2 + self.potential(x))
        self.hamiltonian_deriv_p = kwargs.get('hamiltonian_deriv_p', lambda x, p: p)
        self.F_func = kwargs.get('F_func', lambda x, m: 0.5 * m**2)
        self.F_func_deriv_m = kwargs.get('F_func_deriv_m', lambda x, m: m)

        # Calculate gradient with periodic boundary conditions
        # Make sure gradient has same size as interior points (M-1)

        # Discretization
        self._discretization()

        # Store convergence history
        self.error_history = {'total': [], 'm': [], 'u': []}

        # Setup matrices
        self._setup_basic_matrices()

        # Set default or custom initial/terminal conditions
        self._set_default_conditions()

        # Setup additional matrices
        self._setup_boundary_matrices()

    def _set_default_conditions(self):
        """Set default or custom initial/terminal conditions"""
        if self.m_init is None:
            self.m_init = lambda x: 1 + 0.5 * np.cos(4 * np.pi * x)
        if self.u_term is None:
            self.u_term = lambda x: np.zeros_like(x)

    def _discretization(self):
        # Grid parameters
        self.dt = self.T / self.N
        self.t = np.linspace(0, self.T, self.N + 1)  # time grid has N+1 points

        self.dx = self.X / self.M
        self.x = np.linspace(0, self.X, self.M + 1)  #

        # Initialize solution
        self.u_current = self.u_term(self.x)
        self.u_next = np.zeros_like(self.u_current)
        self.u_prev = np.zeros_like(self.u_current)
        self.u_old = np.zeros_like(self.u_current)
        self.m_current = self.m_init(self.x)
        self.m_next = np.zeros_like(self.m_current)
        self.m_prev = np.zeros_like(self.m_current)
        self.m_old = np.zeros_like(self.m_current)

        # Initialize solution arrays
        # store m and u in matrices of size N+1 x M+1
        self.m_store = np.zeros((self.N + 1, self.M + 1))
        self.u_store = np.zeros((self.N + 1, self.M + 1))
        self.u_store[-2] = [
            0.21486721,
            0.2075614,
            0.1245926,
            0.00281925,
            -0.1184688,
            -0.20901957,
            -0.25166382,
            -0.24016895,
            -0.17612104,
            -0.06958008,
            0.0581872,
            0.17576343,
            0.24951195,
            0.25642903,
            0.1949944,
            0.08583865,
            -0.03821726,
            -0.1448157,
            -0.21039671,
            -0.22266705,
            -0.17972463,
            -0.08999091,
            0.02580829,
            0.13392367,
            0.19237932,
        ]
        self.m_store[1] = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.17474761,
                0.66402542,
                1.46055943,
                2.49867314,
                3.52565227,
                4.17634213,
                4.17634213,
                3.52565227,
                2.49867314,
                1.46055943,
                0.66402542,
                0.17474761,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

    def _setup_basic_matrices(self):
        pass

    def _setup_boundary_matrices(self):
        pass

    @abstractmethod
    def solve(self):
        """Abstract solve method to be implemented by subclasses"""
        pass


class PicardMFGSolver(BaseMFGSolver):
    """Picard iteration solver for Mean Field Games using HJB-FP system."""

    def __init__(
        self,
        N: int,  # time steps
        M: int,  # space steps
        sigma,
        T=1,
        X=1,
        tol=1e-8,
        max_iter=100,
        theta=0.5,  # Damping coefficient: 0 = no update; 1 = full update
        **kwargs,
    ):
        super().__init__(N, M, sigma, T, X, tol, max_iter, **kwargs)
        self.theta = theta

    def _setup_basic_matrices(self):
        """Setup basic matrices for the solver"""
        pass  # We'll handle matrices differently in this implementation

    def solve_forward_m(self, u_current):
        """
        Solve Fokker-Planck equation forward in time
        """
        M = self.M
        dx = self.dx
        dt = self.dt
        sigma = self.sigma

        # Initialize solution array
        m = np.zeros((self.N + 1, M + 1))
        m[0] = self.m_init(self.x)

        for k in range(1, self.N + 1):
            # Get u at time k-1 (semi-implicit)
            u_km1 = u_current[k - 1]

            # Setup matrix elements
            # Below diagonal
            ml = np.zeros(M)
            for i in range(1, M):
                ml[i] = -(sigma**2) / (2 * dx**2)
                ml[i] += -0.5 * np.minimum(u_km1[i] - u_km1[i - 1], 0) / (dx**2)

            # Diagonal
            md = np.zeros(M)
            for i in range(1, M - 1):
                md[i] = sigma**2 / (dx**2) + 1 / dt
                md[i] += (
                    0.5 * (np.maximum(u_km1[i + 1] - u_km1[i], 0) + np.minimum(u_km1[i] - u_km1[i - 1], 0)) / (dx**2)
                )
            md[0] = md[M - 1] = 1.0  # Boundary conditions

            # Above diagonal
            mu = np.zeros(M)
            for i in range(0, M - 1):
                mu[i] = -(sigma**2) / (2 * dx**2)
                mu[i] += -0.5 * np.maximum(u_km1[i + 1] - u_km1[i], 0) / (dx**2)

            # Construct sparse matrix
            ml = np.roll(ml, -1)
            mu = np.roll(mu, 1)

            # Add periodic boundary conditions
            ml_corner = np.zeros(M)
            mu_corner = np.zeros(M)
            ml_corner[M - 1] = -(sigma**2) / (2 * dx**2)
            mu_corner[0] = -(sigma**2) / (2 * dx**2)

            matrix = sp.sparse.spdiags([ml_corner, ml, md, mu, mu_corner], [M - 1, -1, 0, 1, -(M - 1)], M, M).tocsr()

            # RHS vector
            rhs = m[k - 1] / dt

            # Solve system
            m[k] = sp.sparse.linalg.spsolve(matrix, rhs)

        return m

    def solve_backward_u(self, m_current):
        """
        Solve HJB equation backward in time
        """
        M = self.M
        dx = self.dx
        dt = self.dt
        sigma = self.sigma

        # Initialize solution array
        u = np.zeros((self.N + 1, M + 1))
        u[self.N] = self.u_term(self.x)

        for k in range(self.N - 1, -1, -1):
            u_next = u[k + 1]
            m_k = m_current[k]

            # Setup matrix elements similar to forward solve but for backward equation
            # Below diagonal
            ml = np.zeros(M)
            for i in range(1, M):
                ml[i] = -(sigma**2) / (2 * dx**2)
                ml[i] += -0.5 * np.minimum(0, m_k[i - 1]) / dx

            # Diagonal
            md = np.zeros(M)
            for i in range(1, M - 1):
                md[i] = sigma**2 / (dx**2) + 1 / dt + m_k[i] ** 2

            # Above diagonal
            mu = np.zeros(M)
            for i in range(0, M - 1):
                mu[i] = -(sigma**2) / (2 * dx**2)
                mu[i] += 0.5 * np.maximum(0, m_k[i + 1]) / dx

            # Construct sparse matrix similar to forward solve
            matrix = sp.sparse.spdiags([ml, md, mu], [-1, 0, 1], M, M).tocsr()

            # RHS vector includes potential term
            rhs = u_next / dt + self.potential(self.x)

            # Solve system
            u[k] = sp.sparse.linalg.spsolve(matrix, rhs)

        return u

    def solve(self):
        """Main solver using Picard iterations"""
        print("\nSolving MFG using Picard iterations...")

        # Initialize solution arrays
        m_old = np.zeros((self.N + 1, self.M + 1))
        u_old = np.zeros((self.N + 1, self.M + 1))

        # Set initial conditions
        for n in range(self.N + 1):
            m_old[n] = self.m_init(self.x)
            u_old[n] = self.u_term(self.x)

        # Main iteration loop
        for iter in range(self.max_iter):
            # Solve HJB backward
            u_new_tmp = self.solve_backward_u(m_old)
            u_new = self.theta * u_new_tmp + (1 - self.theta) * u_old

            # Solve FP forward
            m_new_tmp = self.solve_forward_m(u_new)
            m_new = self.theta * m_new_tmp + (1 - self.theta) * m_old

            # Compute errors
            err_u = np.linalg.norm(u_new - u_old) * np.sqrt(self.dx * self.dt)
            err_m = np.linalg.norm(m_new - m_old) * np.sqrt(self.dx * self.dt)
            err_total = max(err_u, err_m)

            print(f"Iteration {iter+1}: Error = {err_total:.2e}")

            # Store errors
            self.error_history['total'].append(err_total)
            self.error_history['m'].append(err_m)
            self.error_history['u'].append(err_u)

            # Check convergence
            if err_total < self.tol:
                print(f"\nConverged after {iter+1} iterations")
                break

            # Update solutions
            m_old = m_new.copy()
            u_old = u_new.copy()

        return m_new, u_new


class ConservativeMFGSolver(BaseMFGSolver):
    """Conservative Mean Field Games solver implementing variational discretization"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_basic_matrices()
        self._setup_boundary_matrices()

    def _setup_basic_matrices(self):
        """Setup matrices for interior points"""
        N = self.N
        M = self.M
        dx = self.dx
        dt = self.dt
        sigma = self.sigma

        self.diag_ones = np.eye(M - 1)
        self.shift_right = np.diag([1] * (M - 2), 1)  # upper diagonal
        self.shift_left = np.diag([1] * (M - 2), -1)  # lower diagonal
        self.shift_right2 = np.diag([1] * (M - 3), 2)  # second upper diagonal
        self.shift_left2 = np.diag([1] * (M - 3), -2)  # second lower diagonal

        diag_ones = self.diag_ones
        shift_right = self.shift_right
        shift_right2 = self.shift_right2
        shift_left = self.shift_left
        shift_left2 = self.shift_left2

        # Setup conservative difference matrices

        self.A_u = -1 / (4 * dt) * (diag_ones + shift_right) - sigma / (2 * dx**2) * (
            -2 * diag_ones + shift_right + shift_left
        )

        self.B_u = -1 / (4 * dt) * (-diag_ones + shift_right)

        self.C_u = -1 / (4 * dt) * shift_right - sigma / (2 * dx**2) * (-2 * diag_ones + shift_right + shift_left)

        self.A_m = 1 / (4 * dt) * (diag_ones + shift_right) - sigma / (2 * dx**2) * (
            -2 * diag_ones + shift_right + shift_left
        )

        self.B_m = 1 / (4 * dt) * (-diag_ones + shift_right)  # + csr_matrix(B_m_tilde)

        self.C_m = 1 / (4 * dt) * shift_right - sigma / (2 * dx**2) * (-2 * diag_ones + shift_right + shift_left)

        # Convert to sparse matrices
        self.A_u = csr_matrix(self.A_u)
        self.B_u = csr_matrix(self.B_u)
        self.C_u = csr_matrix(self.C_u)
        self.A_m = csr_matrix(self.A_m)
        self.B_m = csr_matrix(self.B_m)
        self.C_m = csr_matrix(self.C_m)

    @staticmethod
    def expand_matrix_at_position(matrix, new_row=None, new_col=None, row_pos=None, col_pos=None, corner_value=None):
        """
        Expand matrix by adding a row and/or column at specified positions.
        Supports negative indices like NumPy arrays.

        Parameters:
        - matrix: Original numpy array
        - new_row: Row to add (optional)
        - new_col: Column to add (optional)
        - row_pos: Position to add row (optional), supports negative indices
        - col_pos: Position to add column (optional), supports negative indices
        - corner_value: Value for intersection point (if both row and col added)
        """
        rows, cols = matrix.shape
        add_row = new_row is not None and row_pos is not None
        add_col = new_col is not None and col_pos is not None

        # Convert negative indices to positive
        if add_row and row_pos < 0:
            row_pos = rows + 1 + row_pos  # +1 because we're adding to expanded size
        if add_col and col_pos < 0:
            col_pos = cols + 1 + col_pos

        # Determine new dimensions
        new_rows = rows + (1 if add_row else 0)
        new_cols = cols + (1 if add_col else 0)
        expanded = np.zeros((new_rows, new_cols))

        if add_row and add_col:
            # Fill four quadrants
            expanded[:row_pos, :col_pos] = matrix[:row_pos, :col_pos]
            expanded[:row_pos, col_pos + 1 :] = matrix[:row_pos, col_pos:]
            expanded[row_pos + 1 :, :col_pos] = matrix[row_pos:, :col_pos]
            expanded[row_pos + 1 :, col_pos + 1 :] = matrix[row_pos:, col_pos:]

            # Add new row and column
            expanded[row_pos, :col_pos] = new_row[:col_pos]
            expanded[row_pos, col_pos + 1 :] = new_row[col_pos:]
            expanded[:row_pos, col_pos] = new_col[:row_pos]
            expanded[row_pos + 1 :, col_pos] = new_col[row_pos:]

            # Set intersection
            expanded[row_pos, col_pos] = corner_value if corner_value is not None else new_row[col_pos]

        elif add_row:
            # Fill matrix parts
            expanded[:row_pos, :] = matrix[:row_pos, :]
            expanded[row_pos + 1 :, :] = matrix[row_pos:, :]
            # Add new row
            expanded[row_pos, :] = new_row

        elif add_col:
            # Fill matrix parts
            expanded[:, :col_pos] = matrix[:, :col_pos]
            expanded[:, col_pos + 1 :] = matrix[:, col_pos:]
            # Add new column
            expanded[:, col_pos] = new_col

        return expanded

    def _setup_boundary_matrices(self):
        """Setup matrices for the conservative scheme based on boundary condition type"""
        dx = self.dx
        dt = self.dt
        sigma = self.sigma

        # Convert to LIL format for efficient modification
        matrices = {
            'A_u': lil_matrix(self.A_u),
            'B_u': lil_matrix(self.B_u),
            'C_u': lil_matrix(self.C_u),
            'A_m': lil_matrix(self.A_m),
            'B_m': lil_matrix(self.B_m),
            'C_m': lil_matrix(self.C_m),
        }

        # Apply boundary conditions based on type
        if self.boundary.type == 'periodic':

            A_u_row_new_pos = 0
            A_u_row_new = np.zeros(self.M - 1)
            A_u_row_new[0] = -1 / (4 * dt) * 1 - sigma / (2 * dx**2) * (1)

            A_u_col_new_pos = 0
            A_u_col_new = np.zeros(self.M - 1)
            A_u_col_new[0] = -sigma / (2 * dx**2) * (1)
            A_u_col_new[-1] = -1 / (4 * dt) * 1 - sigma / (2 * dx**2) * (1)

            A_u_corner_value = -1 / (4 * dt) * 1 - sigma / (2 * dx**2) * (-2)

            B_u_row_new_pos = 0
            B_u_row_new = np.zeros(self.M - 1)
            B_u_row_new[0] = -1 / (4 * dt) * 1

            B_u_col_new_pos = 0
            B_u_col_new = np.zeros(self.M - 1)
            B_u_col_new[-1] = -1 / (4 * dt) * 1

            B_u_corner_value = -1 / (4 * dt) * (-1)

            C_u_row_new_pos = 0
            C_u_row_new = np.zeros(self.M - 1)
            C_u_row_new[0] = -1 / (4 * dt) * 1 - sigma / (2 * dx**2) * 1

            C_u_col_new_pos = 0
            C_u_col_new = np.zeros(self.M - 1)
            C_u_col_new[0] = -sigma / (2 * dx**2) * 1
            C_u_col_new[-1] = -1 / (4 * dt) * 1 - sigma / (2 * dx**2) * 1

            C_u_corner_value = -sigma / (2 * dx**2) * (-2)

            A_m_row_new_pos = 0
            A_m_row_new = np.zeros(self.M - 1)
            A_m_row_new[0] = 1 / (4 * dt) * 1 - sigma / (2 * dx**2) * 1

            A_m_col_new_pos = 0
            A_m_col_new = np.zeros(self.M - 1)
            A_m_col_new[0] = 1 / (4 * dt) * 1 - sigma / (2 * dx**2) * 2
            A_m_col_new[-1] = 1 / (4 * dt) * 1 - sigma / (2 * dx**2) * 1

            A_m_corner_value = 1 / (4 * dt) * 1 - sigma / (2 * dx**2) * (-2)

            B_m_row_new_pos = 0
            B_m_row_new = np.zeros(self.M - 1)
            B_m_row_new[0] = 1 / (4 * dt) * 1
            B_m_col_new_pos = 0
            B_m_col_new = np.zeros(self.M - 1)
            B_m_col_new[-1] = -1 / (4 * dt) * 1
            B_m_corner_value = -1 / (4 * dt) * 1

            C_m_row_new_pos = 0
            C_m_row_new = np.zeros(self.M - 1)
            C_m_row_new[0] = -sigma / (2 * dx**2) * 1
            C_m_col_new_pos = 0
            C_m_col_new = np.zeros(self.M - 1)
            C_m_col_new[0] = -1 / (4 * dt) * 1 - sigma / (2 * dx**2) * 1

            C_m_corner_value = -sigma / (2 * dx**2) * (-2)

            # Add new row and column to matrices
            matrices['A_u'] = self.expand_matrix_at_position(
                matrices['A_u'].toarray(),
                new_row=A_u_row_new,
                new_col=A_u_col_new,
                row_pos=A_u_row_new_pos,
                col_pos=A_u_col_new_pos,
                corner_value=A_u_corner_value,
            )
            matrices['B_u'] = self.expand_matrix_at_position(
                matrices['B_u'].toarray(),
                new_row=B_u_row_new,
                new_col=B_u_col_new,
                row_pos=B_u_row_new_pos,
                col_pos=B_u_col_new_pos,
                corner_value=B_u_corner_value,
            )

            matrices['C_u'] = self.expand_matrix_at_position(
                matrices['C_u'].toarray(),
                new_row=C_u_row_new,
                new_col=C_u_col_new,
                row_pos=C_u_row_new_pos,
                col_pos=C_u_col_new_pos,
                corner_value=C_u_corner_value,
            )

            matrices['A_m'] = self.expand_matrix_at_position(
                matrices['A_m'].toarray(),
                new_row=A_m_row_new,
                new_col=A_m_col_new,
                row_pos=A_m_row_new_pos,
                col_pos=A_m_col_new_pos,
                corner_value=A_m_corner_value,
            )

            matrices['B_m'] = self.expand_matrix_at_position(
                matrices['B_m'].toarray(),
                new_row=B_m_row_new,
                new_col=B_m_col_new,
                row_pos=B_m_row_new_pos,
                col_pos=B_m_col_new_pos,
                corner_value=B_m_corner_value,
            )

            matrices['C_m'] = self.expand_matrix_at_position(
                matrices['C_m'].toarray(),
                new_row=C_m_row_new,
                new_col=C_m_col_new,
                row_pos=C_m_row_new_pos,
                col_pos=C_m_col_new_pos,
                corner_value=C_m_corner_value,
            )

        elif self.boundary.type == 'dirichlet':
            # Dirichlet boundary conditions
            for mat in matrices.values():
                # Clear boundary rows
                mat[0, :] = mat[-1, :] = 0
                # Set diagonal for direct value assignment
                mat[0, 0] = mat[-1, -1] = 1

        elif self.boundary.type == 'neumann':
            # Neumann boundary conditions (zero gradient)
            for mat in matrices.values():
                # Left boundary
                mat[0, :] = 0
                mat[0, 0] = -3 / (2 * dx)
                mat[0, 1] = 2 / dx
                mat[0, 2] = -1 / (2 * dx)

                # Right boundary
                mat[-1, :] = 0
                mat[-1, -3] = -1 / (2 * dx)
                mat[-1, -2] = 2 / dx
                mat[-1, -1] = -3 / (2 * dx)

        elif self.boundary.type == 'robin':
            # Robin boundary conditions: α∂u/∂n + βu = γ
            if not hasattr(self.boundary, 'left_alpha') or any(
                v is None
                for v in [
                    self.boundary.left_alpha,
                    self.boundary.left_beta,
                    self.boundary.right_alpha,
                    self.boundary.right_beta,
                ]
            ):
                raise ValueError("Robin boundary conditions require alpha and beta coefficients")

            alpha_l, beta_l = self.boundary.left_alpha, self.boundary.left_beta
            alpha_r, beta_r = self.boundary.right_alpha, self.boundary.right_beta

            for mat in matrices.values():
                # Left boundary
                mat[0, :] = 0
                mat[0, 0] = beta_l - 3 * alpha_l / (2 * dx)
                mat[0, 1] = 2 * alpha_l / dx
                mat[0, 2] = -alpha_l / (2 * dx)

                # Right boundary
                mat[-1, :] = 0
                mat[-1, -3] = -alpha_r / (2 * dx)
                mat[-1, -2] = 2 * alpha_r / dx
                mat[-1, -1] = beta_r - 3 * alpha_r / (2 * dx)

        # Convert back to CSR format and store
        self.A_u = matrices['A_u']
        self.B_u = matrices['B_u']
        self.C_u = matrices['C_u']
        self.A_m = matrices['A_m']
        self.B_m = matrices['B_m']
        self.C_m = matrices['C_m']

    @property
    def u_gradient(self):
        """Compute gradient of u at time i using first order differences
        Forward difference at interior points
        Backward difference at last point
        return u_grad: gradient of u at time i
        """
        dx = self.dx
        u_current = self.u_current

        u_grad = np.zeros_like(u_current)
        # Forward differences for all points except last point
        u_grad[:-1] = (u_current[1:] - u_current[:-1]) / dx
        # Backward difference for last point
        u_grad[-1] = (u_current[-1] - u_current[-2]) / dx

        return self._check_numerical_values(u_grad, "u_gradient")

    @property
    def Hamiltonian_deriv_p_current(self):
        """Approximate Hamiltonian derivative with respect to p at time i (current time)
        incuding the values at the boundaries
        hence the size of Hamiltonian_deriv_p_current is M+1

        """
        return self.hamiltonian_deriv_p(self.x, self.u_gradient)

    def _check_numerical_values(self, x, name="value"):
        """Helper method to check for numerical issues"""
        if np.any(np.isnan(x)):
            print(f"WARNING: NaN detected in {name}")
        if np.any(np.abs(x) > 1e10):
            print(f"WARNING: Large values detected in {name}")
        return x

    def _update_B_m_tilde_periodic(self):
        """Update B_m_tilde matrix based on current u values"""
        dx = self.dx
        M = self.M

        # matrices for interior pts, M-1 x M-1
        diag_ones = self.diag_ones
        shift_right = self.shift_right
        shift_right2 = self.shift_right2
        shift_left = self.shift_left
        shift_left2 = self.shift_left2

        # Construct B_m_tilde
        mat1 = diag_ones + shift_left
        mat2 = diag_ones + shift_right

        mat1_new_row_pos = -1
        mat1_new_row = np.zeros(M - 1)
        mat1_new_row[-1] = 1
        mat1_new_col_pos = 0
        mat1_new_col = np.zeros(M - 1)
        mat1_new_col[0] = 1
        mat1_corner_value = 1

        mat2_new_row_pos = -1
        mat2_new_row = np.zeros(M - 1)
        mat2_new_row[0] = 1
        mat2_new_col_pos = 0
        mat2_new_col = np.zeros(M - 1)
        mat2_new_col[-1] = 1
        mat2_corner_value = 1

        mat1 = self.expand_matrix_at_position(
            mat1,
            new_row=mat1_new_row,
            new_col=mat1_new_col,
            row_pos=mat1_new_row_pos,
            col_pos=mat1_new_col_pos,
            corner_value=mat1_corner_value,
        )
        mat2 = self.expand_matrix_at_position(
            mat2,
            new_row=mat2_new_row,
            new_col=mat2_new_col,
            row_pos=mat2_new_row_pos,
            col_pos=mat2_new_col_pos,
            corner_value=mat2_corner_value,
        )

        Hamiltonian_deriv_p_current_1 = np.diag(self.Hamiltonian_deriv_p_current[:-1])
        Hamiltonian_deriv_p_current_2 = np.diag(self.Hamiltonian_deriv_p_current[1:])

        B_m_tilde = 0.5 / dx * (Hamiltonian_deriv_p_current_1 @ mat1 + Hamiltonian_deriv_p_current_2 @ mat2)

        # Update B_m
        self.B_m = (1 / (4 * dx)) * (-np.eye(M) + np.diag([1] * (M - 1), 1)) + B_m_tilde

    def _compute_G_vectors(self):
        """Compute G_u and G_m vectors"""
        dx = self.dx
        M = self.M

        # Get interior points
        # Calculate gradient
        u_grad = self.u_gradient

        # Compute terms
        H_current = self.hamiltonian(self.x, u_grad)
        F_deriv_current = self.F_func_deriv_m(self.x, self.m_current)

        # Construct vectors (both size M)
        self.G_u = np.zeros(M)
        self.G_u[:] = 0.5 * (H_current[1:] + H_current[:-1]) - F_deriv_current[:-1]
        self.G_m = np.zeros(M)

    def solve_forward_m(self):
        """
        Solve forward iteration for m equation:
        m_next = -A_m^{-1}[B_m m_current + C_m m_prev + G_m]
        Solve forward iteration for m equation for interior points
        Matrix operators are (M-1)*(M-1) for interior points [1:M-1]
        """
        # Get sizes
        M = self.M

        # Convert matrices to CSR format if not already
        if not isinstance(self.A_m, csr_matrix):
            self.A_m = csr_matrix(self.A_m)
        if not isinstance(self.B_m, csr_matrix):
            self.B_m = csr_matrix(self.B_m)
        if not isinstance(self.C_m, csr_matrix):
            self.C_m = csr_matrix(self.C_m)
        if not isinstance(self.G_m, csr_matrix):
            self.G_m = csr_matrix(self.G_m)

        # Extract intnp.zeros_like(self.m_current)c boundary conditions use [:-1]
        m_current_calc = np.asarray(self.m_current[:-1]).ravel()
        m_prev_calc = np.asarray(self.m_prev[:-1]).ravel()
        G_m = self.G_m

        # Compute each term separately

        term1 = self.B_m.dot(m_current_calc)
        term2 = self.C_m.dot(m_prev_calc)

        # Compute right-hand side
        rhs = term1 + term2 + G_m
        # Make sure rhs is a vector of length M
        rhs = np.asarray(rhs).ravel()

        # Solve system
        m_next_calc = -spsolve(self.A_m, rhs)
        m_next_calc = np.asarray(m_next_calc).ravel()

        # Create full solution vector with periodic BC
        m_next = np.zeros(M + 1)
        m_next[:-1] = m_next_calc
        m_next[-1] = m_next[0]

        return m_next

    def solve_backward_u(self):
        """
        Solve backward iteration for u equation:
        u_prev = -C_u^{-1}[B_u u_current + A_u u_next + G_u]
        """
        # Convert matrices to CSR format if not already
        if not isinstance(self.C_u, csr_matrix):
            self.C_u = csr_matrix(self.C_u)
        if not isinstance(self.B_u, csr_matrix):
            self.B_u = csr_matrix(self.B_u)
        if not isinstance(self.A_u, csr_matrix):
            self.A_u = csr_matrix(self.A_u)
        if not isinstance(self.G_u, csr_matrix):
            self.G_u = csr_matrix(self.G_u)
        # Extract  points
        u_current_calc = np.asarray(self.u_current[:-1]).ravel()  # Shape should be (M,)
        u_next_calc = np.asarray(self.u_next[:-1]).ravel()  # Shape should be (M,)
        G_u = self.G_u  # Shape should be (M,)

        # Compute right-hand side
        rhs = self.B_u.dot(u_current_calc) + self.A_u.dot(u_next_calc) + G_u
        rhs = np.asarray(rhs).ravel()

        # Solve system
        u_prev_calc = -spsolve(self.C_u, rhs)
        u_prev_calc = np.asarray(u_prev_calc).ravel()

        # Create full solution vector with periodic BC
        u_prev = np.zeros(self.M + 1)
        u_prev[:-1] = u_prev_calc
        u_prev[-1] = u_prev[0]  # Periodic BC

        return u_prev

    def solve(self):
        """Main solver loop implementing the backward-forward scheme"""

        # Initialize terminal and initial conditions
        self.m_store[0] = self.m_init(self.x)  # Initial condition for m
        self.u_store[-1] = self.u_term(self.x)  # Terminal condition for u

        # Initialize solution arrays for the iterative process
        m_old = np.copy(self.m_store)  # Full space-time array
        u_old = np.copy(self.u_store)  # Full space-time array

        # Initialize progress bars

        iter_progress = tqdm(total=self.max_iter, desc='Fixed Point Iterations', position=0)
        forward_progress = tqdm(total=self.N, desc='Forward solve', position=1)
        backward_progress = tqdm(total=self.N, desc='Backward solve', position=2)
        print("\n" * 3)

        # Main iteration loop with tqdm

        try:
            for iter in range(self.max_iter):
                # Reset sub-progress bars
                forward_progress.reset()
                backward_progress.reset()
                # Forward solve for m
                for i in range(1, self.N):
                    # Set current and previous values for this time step
                    self.m_current = m_old[i]
                    self.m_prev = m_old[i - 1]
                    self.u_current = u_old[-i - 1]

                    # Update B_m_tilde based on current u gradient
                    self._update_B_m_tilde_periodic()

                    # Compute G vectors for this time step
                    self._compute_G_vectors()

                    # Forward iteration for m
                    m_next = self.solve_forward_m()
                    self.m_store[i + 1] = m_next

                    # Update forward progress
                    forward_progress.update(1)
                    forward_progress.refresh()

                # Backward solve for u
                for j in range(self.N - 1, 0, -1):
                    # Set current and next values for this time step
                    self.u_current = u_old[j]
                    self.u_next = self.u_store[j + 1]  # Use latest computed values
                    self.m_current = self.m_store[self.N - j]
                    # Use latest computed values

                    # Compute G vectors for this time step
                    self._compute_G_vectors()

                    # Backward iteration for u
                    u_prev = self.solve_backward_u()
                    self.u_store[j - 1] = u_prev

                    # Update backward progress
                    backward_progress.update(1)
                    backward_progress.refresh()

                # Compute errors using full space-time arrays
                err_m = np.max(np.abs(self.m_store - m_old))
                err_u = np.max(np.abs(self.u_store - u_old))
                err_total = max(err_m, err_u)

                # Update progress information
                iter_progress.update(1)

                # Update progress bar descriptions with error info
                iter_progress.set_description(f'Iteration {iter+1}/{self.max_iter}')
                iter_progress.set_postfix(
                    {'Error': f'{err_total:.2e}', 'm_err': f'{err_m:.2e}', 'u_err': f'{err_u:.2e}'}
                )
                iter_progress.refresh()
                # Store errors
                self.error_history['total'].append(err_total)
                self.error_history['m'].append(err_m)
                self.error_history['u'].append(err_u)

                # Check convergence
                if err_total < self.tol:
                    print(f"\nConverged after {iter+1} iterations")
                    break

                # Update old solutions for next iteration
                m_old = np.copy(self.m_store)
                u_old = np.copy(self.u_store)

            else:
                print(f"\nWarning: Maximum iterations ({self.max_iter}) reached without convergence")

                # Close progress bars
        finally:
            # Ensure all progress bars are closed
            iter_progress.close()
            forward_progress.close()
            backward_progress.close()
            print("\n")

            # Add some spacing after progress bars

        return self.m_store, self.u_store

    def analyze_solution(self):
        """
        plots relevant metrics.
        """
        pass


# Example usage
if __name__ == "__main__":

    # Parameters
    sigma = 0.1
    T = 1.0
    X = 1.0
    tol = 1e-6
    max_iter = 10
    bc = BoundaryCondition(type='periodic')

    # Define custom functions
    def potential(x):
        return np.sin(2 * np.pi * x)

    def hamiltonian(x, p):
        return 0.5 * np.square(p) + potential(x)

    def hamiltonian_deriv_p(x, p):
        '''
        return the derivative of the hamiltonian with respect to p
        including the values at the boundaries
        hence the size of returned value is M+1
        '''
        return p

    def F_func(x, m):
        return 0.5 * m**2

    def F_func_deriv_m(x, m):
        return m

    def m_init(x):
        return 1 + 0.5 * np.cos(4 * np.pi * x)

    def u_term(x):
        return np.zeros_like(x)

    # Create solver with custom functions
    solver = ConservativeMFGSolver(
        N=100,  # time steps
        M=24,  # space steps
        sigma=0.01,
        T=1.0,
        X=1.0,
        boundary=bc,
        potential=potential,
        hamiltonian=hamiltonian,
        hamiltonian_deriv_p=hamiltonian_deriv_p,
        F_func=F_func,
        F_func_deriv_m=F_func_deriv_m,
        m_init=m_init,
        u_term=u_term,
        tol=tol,
        max_iter=max_iter,
    )

    # Solve
    m, u = solver.solve()

    # Analyze solution
    # conservation_metrics, energy_metrics = solver.analyze_solution()
