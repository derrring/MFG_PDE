import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Tuple

# Import npart and ppart from the utils module
from mfg_pde.utils.aux_func import npart, ppart

# Define a limit for values before squaring to prevent overflow within H
VALUE_BEFORE_SQUARE_LIMIT = 1e150


class MFGProblem(ABC):
    def __init__(
        self,
        xmin: float = 0.0,
        xmax: float = 1.0,
        Nx: int = 51,
        T: float = 1.0,
        Nt: int = 51,
        sigma: float = 1.0,
        coefCT: float = 0.5,
        **kwargs: Any,
    ) -> None:

        self.xmin: float = xmin
        self.xmax: float = xmax
        self.Lx: float = xmax - xmin
        self.Nx: int = Nx
        self.Dx: float = (xmax - xmin) / Nx if Nx > 0 else 0.0

        self.T: float = T
        self.Nt: int = Nt
        self.Dt: float = T / Nt if Nt > 0 else 0.0

        self.xSpace: np.ndarray = np.linspace(xmin, xmax, Nx + 1, endpoint=True)
        self.tSpace: np.ndarray = np.linspace(0, T, Nt + 1, endpoint=True)

        self.sigma: float = sigma
        self.coefCT: float = coefCT

        self.f_potential: np.ndarray
        self.u_fin: np.ndarray
        self.m_init: np.ndarray
        self._initialize_functions(**kwargs)

    def _potential(self, x: float) -> float:
        return 50 * (
            0.1 * np.cos(x * 2 * np.pi / self.Lx)
            + np.cos(x * 4 * np.pi / self.Lx)
            + 0.1 * np.sin((x - np.pi / 8) * 2 * np.pi / self.Lx)
        )

    def _initialize_functions(self, **kwargs: Any) -> None:
        self.f_potential = np.zeros(self.Nx + 1)
        self.u_fin = np.zeros(self.Nx + 1)
        self.m_init = np.zeros(self.Nx + 1)

        for i in range(self.Nx + 1):
            x_i = self.xSpace[i]

            self.f_potential[i] = self._potential(x_i)
            self.u_fin[i] = kwargs.get("g_final_func", lambda x_val: 0.0)(x_i)

            default_m0_func = lambda x_val: np.exp(
                -np.power(x_val - self.Lx / 2.0, 2.0)
                / (2 * np.power(self.Lx / 10.0, 2.0))
            )
            m_init_i = kwargs.get("m_initial_func", default_m0_func)(x_i)
            self.m_init[i] = max(m_init_i - 0.05, 0)

        # Always normalize initial condition m_0 to be a probability density (integral = 1)
        # This ensures m_0 is a proper initial distribution, but FDM evolution is natural
        if np.sum(self.m_init) * self.Dx > 1e-9:
            self.m_init /= np.sum(self.m_init) * self.Dx
        elif self.Nx > 0:
            self.m_init = np.ones(self.Nx + 1) / (
                (self.Nx + 1) * self.Dx if self.Dx > 0 else (self.Nx + 1)
            )

    @abstractmethod
    def H(
        self,
        x_idx: int,
        m_at_x: float,
        p_values: Dict[str, float],
        t_idx: Optional[int] = None,
    ) -> float:
        pass

    @abstractmethod
    def dH_dm(
        self,
        x_idx: int,
        m_at_x: float,
        p_values: Dict[str, float],
        t_idx: Optional[int] = None,
    ) -> float:
        pass

    def get_hjb_hamiltonian_jacobian_contrib(
        self,
        U_for_jacobian_terms: np.ndarray,  # This will be U_k_n (from prev. Picard) for ExampleMFGProblem
        t_idx_n: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return None

    def get_hjb_residual_m_coupling_term(
        self,
        M_density_at_n_plus_1: np.ndarray,
        U_n_current_guess_derivatives: Dict[str, np.ndarray],
        x_idx: int,
        t_idx_n: int,
    ) -> Optional[float]:
        return None

    def get_final_u(self) -> np.ndarray:
        return self.u_fin.copy()

    def get_initial_m(self) -> np.ndarray:
        return self.m_init.copy()


class ExampleMFGProblem(MFGProblem):
    def H(
        self,
        x_idx: int,
        m_at_x: float,
        p_values: Dict[str, float],
        t_idx: Optional[int] = None,
    ) -> float:
        p_forward = p_values.get("forward")
        p_backward = p_values.get("backward")

        if p_forward is None or p_backward is None:
            return np.nan
        if (
            np.isnan(p_forward)
            or np.isinf(p_forward)
            or np.isnan(p_backward)
            or np.isinf(p_backward)
            or np.isnan(m_at_x)
            or np.isinf(m_at_x)
        ):
            return np.nan

        npart_val_fwd = npart(p_forward)
        ppart_val_bwd = ppart(p_backward)

        if (
            abs(npart_val_fwd) > VALUE_BEFORE_SQUARE_LIMIT
            or abs(ppart_val_bwd) > VALUE_BEFORE_SQUARE_LIMIT
        ):
            return np.nan

        try:
            term_npart_sq = npart_val_fwd**2
            term_ppart_sq = ppart_val_bwd**2
        except OverflowError:
            return np.nan

        if (
            np.isinf(term_npart_sq)
            or np.isnan(term_npart_sq)
            or np.isinf(term_ppart_sq)
            or np.isnan(term_ppart_sq)
        ):
            return np.nan

        hamiltonian_control_part = (0.5) * self.coefCT * (term_npart_sq + term_ppart_sq)

        if np.isinf(hamiltonian_control_part) or np.isnan(hamiltonian_control_part):
            return np.nan

        potential_cost_V_x = self.f_potential[x_idx]
        try:
            coupling_cost_with_m = m_at_x**2
        except OverflowError:
            return np.nan

        if (
            np.isinf(potential_cost_V_x)
            or np.isnan(potential_cost_V_x)
            or np.isinf(coupling_cost_with_m)
            or np.isnan(coupling_cost_with_m)
        ):
            return np.nan

        result = hamiltonian_control_part - potential_cost_V_x - coupling_cost_with_m

        if np.isinf(result) or np.isnan(result):
            return np.nan

        return result

    def dH_dm(
        self,
        x_idx: int,
        m_at_x: float,
        p_values: Dict[str, float],
        t_idx: Optional[int] = None,
    ) -> float:
        if np.isnan(m_at_x) or np.isinf(m_at_x):
            return np.nan
        return 2 * m_at_x

    def get_hjb_hamiltonian_jacobian_contrib(
        self,
        U_for_jacobian_terms: np.ndarray,  # This is Uoldn from notebook (U from prev Picard iter)
        t_idx_n: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        Nx = self.Nx + 1
        Dx = self.Dx
        coefCT = self.coefCT

        J_D_H = np.zeros(Nx)
        J_L_H = np.zeros(Nx)
        J_U_H = np.zeros(Nx)

        if abs(Dx) < 1e-14 or Nx <= 1:
            return J_D_H, J_L_H, J_U_H

        U_curr = U_for_jacobian_terms  # Use the U from previous Picard iteration for these terms

        for i in range(Nx):
            ip1 = (i + 1) % Nx
            im1 = (i - 1 + Nx) % Nx

            # Derivatives of U_curr (U from previous Picard iteration)
            p1_i = (U_curr[ip1] - U_curr[i]) / Dx  # p_forward at i using U_curr
            p2_i = (U_curr[i] - U_curr[im1]) / Dx  # p_backward at i using U_curr

            # Terms from the original notebook's Jacobian (_getJacobianU -> MD, ML, MU for H-part)
            J_D_H[i] = coefCT * (npart(p1_i) + ppart(p2_i)) / (Dx**2)
            J_L_H[i] = -coefCT * ppart(p2_i) / (Dx**2)
            J_U_H[i] = -coefCT * npart(p1_i) / (Dx**2)

        return J_D_H, J_L_H, J_U_H

    def get_hjb_residual_m_coupling_term(
        self,
        M_density_at_n_plus_1: np.ndarray,
        U_n_current_guess_derivatives: Dict[str, np.ndarray],
        x_idx: int,
        t_idx_n: int,
    ) -> Optional[float]:
        m_val = M_density_at_n_plus_1[x_idx]
        if np.isnan(m_val) or np.isinf(m_val):
            return np.nan
        try:
            term = -2 * (m_val**2)  # This was mdmH_withM, which was ADDED to residual
        except OverflowError:
            return np.nan
        if np.isinf(term) or np.isnan(term):
            return np.nan
        return term
