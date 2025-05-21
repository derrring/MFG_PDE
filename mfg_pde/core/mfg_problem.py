import numpy as np
import scipy as sc
import scipy.sparse as sparse
import scipy.sparse.linalg
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

class MFGProblem:
    def __init__(self, xmin=0.0, xmax=1.0, Nx=51, T=0.5, Nt=51, sigma=1.0, coefCT=0.5):
        # Grid and Time Parameters
        self.xmin = xmin
        self.xmax = xmax
        self.Lx = xmax - xmin
        self.Nx = Nx #TOTAL number of spatial knots
        self.Dx = (xmax - xmin) / (Nx-1)
        self.T = T
        self.Nt = Nt #TOTAL number of temporal knots
        self.Dt = T / (Nt-1)
        self.xSpace = np.linspace(xmin, xmax, Nx, endpoint=True) # num include the endpoint
        self.tSpace = np.linspace(0, T, Nt, endpoint=True)
        self.sigma = sigma
        self.coefCT = coefCT

        # Problem Specific Functions (Costs, Initial Density, Hamiltonian)
        self._initialize_functions()

    def _potential(self, x):
        """Running cost potential function F(x,m) = potential(x) + m^2"""
        return 50 * (0.1 * np.cos(x * 2 * np.pi) + np.cos(x * 4 * np.pi) + 0.1 * np.sin((x - np.pi / 8) * 2 * np.pi))

    def _initialize_functions(self):
        """Initializes cost functions and initial density"""
        self.f_potential = np.zeros(self.Nx) # Potential part of running cost F
        self.u_fin = np.zeros(self.Nx) # Final cost g
        self.m_init = np.zeros(self.Nx) # Initial density m0

        for i in range(self.Nx):
            x_i = self.xmin + i * self.Dt
            # Running cost potential F(x)
            self.f_potential[i] = self._potential(x_i)
            # Final cost G(x) = 0 in this example
            self.u_fin[i] = 0
            # Initial Density m0(x)
            m_init_i = np.exp(-np.power(x_i - self.Lx / 2., 2.) / (2 * np.power((self.xmax - self.xmin) / 10., 2.)))
            self.m_init[i] = max(m_init_i - 0.05, 0) # Truncate for BC=0

        self.m_init /= sum(self.m_init * self.Dx) # Normalize initial density

    def H(self, m, i, p1, p2):
        """Hamiltonian H(x, p, m) used in HJB"""
        # Note: The original H_withM included the F term (-f[i] - m[i]**2)
        # It's often cleaner to separate H(x,p) and F(x,m)
        # Even for non-separable H(x,p,m) and F(x,m)
        # H(x,p) = 0.5 * coefCT * (npart(p1)**2 + ppart(p2)**2)
        # F(x,m) = self.f_potential[i] + m[i]**2
        # The HJB equation involves -H(x, Du) - F(x,m)
        # The original implementation combines these in H_withM
        hamiltonian_p = (1./2.) * self.coefCT * (self._npart(p1)**2 + self._ppart(p2)**2)
        # We keep the original structure where H_withM = H(p) - F(x,m)
        return hamiltonian_p - self.f_potential[i] - m[i]**2

    def dH_dm(self, m, i, p1, p2):
            """Derivative of Hamiltonian H(x, p, m) w.r.t m (for coupling)"""
            # Corresponds to the coupling term G(m) or F(x,m). Here: -d(m^2)/dm = -2m
            # Let's assume the original intent was F(x,m) = f_potential(x) + m^2
            # Then dF/dm = 2*m[i]
            # The HJB eq is -du/dt - H(x, Du) - F(x,m) = sigma^2/2 * d^2u/dx^2
            # For clarity, let's return the derivative of the F term wrt m: dF/dm = 2*m[i]
            return 2 * m[i] # Derivative of F w.r.t m if F(x,m) = f_potential + m^2

    # --- Helper functions from original code ---
    def _ppart(self, x):
        return np.maximum(x, 0)

    def _npart(self, x):
        return -np.minimum(x, 0)

    def get_final_u(self):
        """Returns the final condition for U"""
        return self.u_fin

    def get_initial_m(self):
        """Returns the initial condition for M"""
        return self.m_init

