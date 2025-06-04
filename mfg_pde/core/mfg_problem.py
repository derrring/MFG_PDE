import numpy as np
import scipy as sc  # Not strictly used in this file, but often in the ecosystem
import scipy.sparse as sparse  # Not strictly used
import scipy.sparse.linalg  # Not strictly used
import scipy.interpolate as interpolate  # Not strictly used
import matplotlib.pyplot as plt  # Not strictly used

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any


class MFGProblem(ABC):
    """
    Abstract Base Class for defining a Mean Field Game problem.

    This class sets up the grid, time parameters, and basic cost structures.
    The Hamiltonian H(x, p, m) and its derivative dH/dm must be implemented
    by concrete subclasses.
    """

    def __init__(
        self,
        xmin: float = 0.0,
        xmax: float = 1.0,
        Nx: int = 51, # TOTAL number of spatial knots
        T: float = 1,
        Nt: int = 51, # TOTAL number of temporal knots (time points)
        sigma: float = 1.0,
        coefCT: float = 0.5, # Coefficient for control/drift term in Hamiltonian/SDE
        **kwargs: Any,
    ) -> None:
        
        # Grid and Time Parameters
        self.xmin: float = xmin
        self.xmax: float = xmax
        self.Lx: float = xmax - xmin
        self.Nx: int = Nx  
        self.Dx: float = (xmax - xmin) / (Nx - 1) if Nx > 1 else 0.0 # Dx for Nx-1 intervals

        self.T: float = T
        self.Nt: int = Nt  
        self.Dt: float = T / (Nt - 1) if Nt > 1 else 0.0  # Dt for Nt-1 intervals

        self.xSpace: np.ndarray = np.linspace(xmin, xmax, Nx, endpoint=True)
        self.tSpace: np.ndarray = np.linspace(0, T, Nt, endpoint=True)

        self.sigma: float = sigma  # Diffusion coefficient for HJB/FP
        self.coefCT: float = (
            coefCT  # Coefficient for control/drift term in Hamiltonian/SDE, i.e. b(x,m) = coefCT * m
        )

        # Problem Specific Functions (Costs, Initial Density)
        # These are initialized by default but can be overridden by subclasses if needed.
        self.f_potential: (
            np.ndarray
        )  # Running cost potential F(x) (excluding m-dependent part)
        self.u_fin: np.ndarray  # Final cost g(x) for U(T,x)
        self.m_init: np.ndarray  # Initial density m0(x) for M(0,x)
        self._initialize_functions(**kwargs)

    def _potential(self, x: float) -> float:
        """
        Default running cost potential function component V(x).
        The full running cost F(x,m) is often V(x) + coupling_term(m).
        This can be overridden by subclasses.
        """
        # Example: 50 * (0.1 * cos(x*2pi) + cos(x*4pi) + 0.1 * sin((x-pi/8)*2pi))
        return 50 * (
            0.1 * np.cos(x * 2 * np.pi / self.Lx)  # Scale to domain length Lx
            + np.cos(x * 4 * np.pi / self.Lx)
            + 0.1 * np.sin((x - np.pi / 8) * 2 * np.pi / self.Lx)
        )

    def _initialize_functions(self, **kwargs: Any) -> None:
        """
        Initializes default cost functions and initial density.
        Subclasses can override this or parts of it.
        """
        self.f_potential = np.zeros(self.Nx)
        self.u_fin = np.zeros(self.Nx)
        self.m_init = np.zeros(self.Nx)

        for i in range(self.Nx):
            x_i = self.xSpace[i]

            # Running cost potential V(x) (part of F(x,m))
            self.f_potential[i] = self._potential(x_i)

            # Final cost G(x) = 0 in this default example
            self.u_fin[i] = kwargs.get("g_final_func", lambda x_val: 0.0)(x_i)

            # Initial Density m0(x)
            # Default: Gaussian centered, truncated, and normalized
            default_m0_func = lambda x_val: np.exp(
                -np.power(x_val - self.Lx / 2.0, 2.0)
                / (2 * np.power(self.Lx / 10.0, 2.0))
            )
            m_init_i = kwargs.get("m_initial_func", default_m0_func)(x_i)
            self.m_init[i] = max(m_init_i - 0.05, 0)  # Example truncation

        if (
            np.sum(self.m_init) * self.Dx > 1e-9
        ):  # Avoid division by zero if m_init is all zero
            self.m_init /= np.sum(self.m_init) * self.Dx  # Normalize initial density
        else:
            # Handle case of zero initial mass, e.g., uniform small density
            # print("Warning: Initial mass is close to zero. Setting to uniform.")
            self.m_init = np.ones(self.Nx) / (self.Nx * self.Dx)

    @abstractmethod
    def H(
        self,
        x_idx: int,
        m_at_x: float,
        p_values: Dict[str, float],
        t_idx: Optional[int] = None,
    ) -> float:
        """
        Abstract method for the Hamiltonian H(x, m, p, t).
        This function is used in the HJB equation.
        The HJB equation is typically of the form:
        -U_t - sigma^2/2 * U_xx + H(x, m, Du, t) = 0  (for backward HJB)
        or U_t - sigma^2/2 * U_xx + H(x, m, Du, t) = 0 (if H includes -H_control)

        Args:
        x_idx (int): Spatial index i, corresponding to xSpace[i].
        m_at_x (float): Density m(x_i, t_k) at the current point.
        p_values (Dict[str, float]): Dictionary of momentum approximations at x_i, t_k.
                                        Keys might include 'forward', 'backward', 'centered', etc.,
                                        representing different finite difference approximations of Du.
                                        Example: p_values={'forward': (U[i+1]-U[i])/Dx,
                                                        'backward': (U[i]-U[i-1])/Dx}
        t_idx (Optional[int]): Temporal index k, corresponding to tSpace[k]. Can be None.

        Returns:
            float: Value of the Hamiltonian.
        """
        pass

    @abstractmethod
    def dH_dm(
        self,
        x_idx: int,
        m_at_x: float,
        p_values: Dict[str, float],
        t_idx: Optional[int] = None,
    ) -> float:
        """
        Abstract method for the derivative of the Hamiltonian H with respect to m, dH/dm.
        This term is often part of the coupling in the MFG system, appearing in the FP equation
        or as part of the linearization if H itself depends on m.

        Args:
            x_idx (int): Spatial index i.
            m_at_x (float): Density m(x_i, t_k).
            p_values (Dict[str, float]): Dictionary of momentum approximations at x_i, t_k.
            t_idx (Optional[int]): Temporal index k.

        Returns:
            float: Value of dH/dm.
        """
        pass

    def get_final_u(self) -> np.ndarray:
        """Returns the final condition for U (value function at T)."""
        return self.u_fin.copy()  # Return a copy

    def get_initial_m(self) -> np.ndarray:
        """Returns the initial condition for M (density at t=0)."""
        return self.m_init.copy()  # Return a copy


class ExampleMFGProblem(MFGProblem):
    """
    A concrete implementation of MFGProblem, demonstrating the original
    Hamiltonian structure using _npart and _ppart.
    H(x,m,p) = 0.5 * coefCT * (npart(p_fwd)^2 + ppart(p_bwd)^2) - V(x) - m^2
    """

    def _ppart(self, x: float) -> float:
        return np.maximum(x, 0.0)

    def _npart(self, x: float) -> float:
        # Original definition was -np.minimum(x,0), which is equivalent to np.maximum(-x,0)
        # For consistency with ppart(x) = max(x,0), npart(x) = max(-x,0)
        # If p_fwd = (U[i+1]-U[i])/Dx, then npart(p_fwd) is for flow to the right if p_fwd < 0.
        # If p_bwd = (U[i]-U[i-1])/Dx, then ppart(p_bwd) is for flow to the left if p_bwd > 0.
        return np.maximum(-x, 0.0)

    def H(
        self,
        x_idx: int,
        m_at_x: float,
        p_values: Dict[str, float],
        t_idx: Optional[int] = None,
    ) -> float:
        """
        Implements the specific Hamiltonian:
        H_control(p_fwd, p_bwd) - V(x) - coupling_m(m)
        where H_control is 0.5 * coefCT * (npart(p_fwd)^2 + ppart(p_bwd)^2),
        V(x) is self.f_potential[x_idx], and coupling_m(m) is m_at_x**2.

        Expects 'forward' and 'backward' keys in p_values.
        p_values['forward'] corresponds to (U[i+1] - U[i]) / Dx
        p_values['backward'] corresponds to (U[i] - U[i-1]) / Dx
        """
        p_forward = p_values.get("forward")
        p_backward = p_values.get("backward")

        if p_forward is None or p_backward is None:
            raise ValueError(
                "Hamiltonian requires 'forward' and 'backward' momentum values in p_values."
            )

        # This specific Hamiltonian form is common in upwind schemes for optimal control
        # H_control = 0.5 * self.coefCT * (self._npart(p_forward)**2 + self._ppart(p_backward)**2)
        # A more standard separable Hamiltonian H(x,p) = 0.5 * p^2 + V(x) would use a centered p.
        # The provided form is often H(x, Du) = sup_alpha {-L(x,alpha) - alpha . Du}
        # If alpha_star = - Du (for L = 0.5 alpha^2), then H = 0.5 (Du)^2.
        # The npart/ppart form implicitly chooses control based on sign of derivatives.

        # For this specific example, let's assume the control part is:
        # 0.5 * coefCT * ( (min(0, p_forward))^2 + (max(0, p_backward))^2 )
        # which is NOT the same as original npart/ppart if npart(x) = -min(x,0)
        # Original: npart(p1)^2 + ppart(p2)^2
        # npart(x) = -min(x,0) --> npart(x)^2 = min(x,0)^2
        # ppart(x) = max(x,0) --> ppart(x)^2 = max(x,0)^2
        # So, H_control = 0.5 * coefCT * ( (min(0, p_forward))**2 + (max(0, p_backward))**2 )
        # This is a common form for upwind discretizations of H = 0.5 |Du|^2 if Du is split.

        # Let's use the definition from the original code's H for this example:
        # hamiltonian_p = (1./2.) * self.coefCT * (self._npart(p1)**2 + self._ppart(p2)**2)
        # where p1 was forward diff, p2 was backward diff.
        # So, p1 = p_forward, p2 = p_backward.

        hamiltonian_control_part = (
            (0.5)
            * self.coefCT
            * (self._npart(p_forward) ** 2 + self._ppart(p_backward) ** 2)
        )

        # Running cost V(x) (potential part)
        potential_cost_V_x = self.f_potential[x_idx]

        # Coupling term with m
        coupling_cost_with_m = m_at_x**2

        # The HJB equation is often -U_t - sigma^2/2 U_xx + H_control(Du) - F(x,m) = 0
        # where F(x,m) = V(x) + coupling_cost_with_m(m).
        # So, the H passed to the solver should be H_control(Du) - V(x) - coupling_cost_with_m(m).
        return hamiltonian_control_part - potential_cost_V_x - coupling_cost_with_m

    def dH_dm(
        self,
        x_idx: int,
        m_at_x: float,
        p_values: Dict[str, float],
        t_idx: Optional[int] = None,
    ) -> float:
        """
        Derivative of H = H_control - V(x) - m^2 with respect to m.
        Only the -m^2 term depends on m.
        So, dH/dm = -2*m.
        """
        # In the original code, dH_dm was given as 2*m. This implies that the coupling term
        # in the HJB was + F(x,m) or the definition of H included -F(x,m) and dH/dm was
        # actually -dF/dm.
        # If HJB is -U_t + H(x,Du) - F(x,m) = Lu, then dH/dm (from HJB perspective) is -dF/dm.
        # If F(x,m) = V(x) + m^2, then -dF/dm = -2*m.
        # The original MFG-FDM-particle2.py had problem.H returning H_control - V(x) - m^2
        # and problem.dH_dm returning 2*m. This is inconsistent if dH_dm is literal.
        # It's more likely that dH_dm was intended to be the derivative of the coupling term F_m(m) = m^2,
        # which is 2m. This term then appears in the FP equation's drift.
        # Let's assume dH_dm is asking for d(CouplingCost)/dm from the perspective of the FP equation.
        # If the coupling cost in F(x,m) is m^2, its derivative is 2*m.
        return 2 * m_at_x


if __name__ == "__main__":
    # Example of how to use the new structure
    print("--- Example MFGProblem Instantiation ---")

    # Create an instance of the concrete problem
    example_problem = ExampleMFGProblem(
        xmin=0.0, xmax=1.0, Nx=11, T=0.1, Nt=11, sigma=0.1, coefCT=0.5
    )

    print(
        f"Domain: [{example_problem.xmin}, {example_problem.xmax}], Nx={example_problem.Nx}, Dx={example_problem.Dx:.2f}"
    )
    print(
        f"Time: [0, {example_problem.T}], Nt={example_problem.Nt}, Dt={example_problem.Dt:.3f}"
    )
    print(
        f"Initial M sum: {np.sum(example_problem.get_initial_m()) * example_problem.Dx:.2f}"
    )  # Should be approx 1.0

    # Example call to H (requires dummy p_values)
    # In a real solver, p_values would be computed from U
    dummy_p_values = {"forward": 0.1, "backward": -0.1}
    x_idx_test = example_problem.Nx // 2
    m_test = example_problem.get_initial_m()[x_idx_test]

    try:
        h_val = example_problem.H(
            x_idx=x_idx_test, m_at_x=m_test, p_values=dummy_p_values
        )
        print(
            f"Example H({example_problem.xSpace[x_idx_test]:.2f}, m={m_test:.2f}, p_fwd={dummy_p_values['forward']}, p_bwd={dummy_p_values['backward']}) = {h_val:.2f}"
        )

        dhdm_val = example_problem.dH_dm(
            x_idx=x_idx_test, m_at_x=m_test, p_values=dummy_p_values
        )
        print(
            f"Example dH/dm({example_problem.xSpace[x_idx_test]:.2f}, m={m_test:.2f}) = {dhdm_val:.2f}"
        )

    except ValueError as e:
        print(f"Error during example H call: {e}")

    # Test abstractness (this should fail if MFGProblem is directly instantiated)
    try:
        # generic_problem = MFGProblem() # This should raise TypeError
        print("MFGProblem (ABC) should not be instantiable directly.")
    except TypeError as e:
        print(f"Correctly caught error for ABC instantiation: {e}")

    # Plot initial density
    plt.figure()
    plt.plot(
        example_problem.xSpace,
        example_problem.get_initial_m(),
        label="Initial Density m(0,x)",
    )
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title("Initial Density from ExampleMFGProblem")
    plt.legend()
    plt.grid(True)
    plt.show()
