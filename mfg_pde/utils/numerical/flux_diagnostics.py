"""
Flux Diagnostics for Fokker-Planck Mass Conservation Analysis.

This module provides tools to compute and analyze boundary fluxes in FP equations,
enabling detailed mass conservation diagnostics beyond simple total mass tracking.

For the Fokker-Planck equation:
    ∂m/∂t + ∇·J = 0

where the probability flux is:
    J = αm - D∇m  (advection + diffusion)

Mass conservation requires:
    ∫_Ω ∂m/∂t dx = -∮_∂Ω J·n dS = 0  (for no-flux BC)

This module computes:
1. Boundary flux at each domain face
2. Total net flux (should be 0 for conservation)
3. Per-timestep flux history
4. Flux breakdown (advection vs diffusion components)

Usage:
    from mfg_pde.utils.numerical.flux_diagnostics import FluxDiagnostics

    # Create diagnostics object
    diag = FluxDiagnostics(dimension=2, spacing=(dx, dy))

    # Compute flux at each timestep
    for t in range(Nt):
        flux_info = diag.compute_boundary_flux(
            m=M[t],
            velocity_field=alpha,  # or gradient of u
            diffusion=sigma**2/2
        )
        print(f"Net flux: {flux_info.net_flux:.2e}")

    # Get summary
    summary = diag.get_summary()
    print(f"Max mass drift: {summary['max_mass_drift_percent']:.2f}%")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class BoundaryFluxResult:
    """Result of boundary flux computation at a single timestep.

    Attributes:
        total_mass: Total mass in domain (integral of m)
        boundary_fluxes: Dict mapping boundary name to flux value
            Positive = outward (leaving domain)
        net_flux: Total net flux out of domain (sum of all boundaries)
        advection_flux: Advection component of net flux (αm term)
        diffusion_flux: Diffusion component of net flux (-D∇m term)
        is_conserved: Whether net_flux ≈ 0 within tolerance
    """

    total_mass: float
    boundary_fluxes: dict[str, float]
    net_flux: float
    advection_flux: float
    diffusion_flux: float
    is_conserved: bool
    tolerance: float = 1e-10


@dataclass
class FluxSummary:
    """Summary of flux diagnostics over entire simulation.

    Attributes:
        initial_mass: Mass at t=0
        final_mass: Mass at t=T
        mass_drift_absolute: |final - initial|
        mass_drift_percent: |final - initial| / initial * 100
        max_net_flux: Maximum |net_flux| over all timesteps
        mean_net_flux: Mean |net_flux| over all timesteps
        flux_history: List of BoundaryFluxResult per timestep
        is_conservative: Whether mass is well-conserved
        integrated_boundary_flux: Total mass lost through boundaries (from flux integration)
        algorithmic_leak: mass change not explained by boundary flux (discretization error)
        algorithmic_leak_percent: algorithmic leak as percentage of initial mass
    """

    initial_mass: float
    final_mass: float
    mass_drift_absolute: float
    mass_drift_percent: float
    max_net_flux: float
    mean_net_flux: float
    flux_history: list[BoundaryFluxResult]
    is_conservative: bool
    boundary_breakdown: dict[str, dict[str, float]]  # per-boundary statistics
    # New fields for separating physical vs algorithmic mass loss
    integrated_boundary_flux: float = 0.0  # sum of net_flux * dt
    algorithmic_leak: float = 0.0  # actual change - integrated flux
    algorithmic_leak_percent: float = 0.0  # as percentage of initial mass


class FluxDiagnostics:
    """Flux diagnostics for Fokker-Planck mass conservation analysis.

    Computes boundary fluxes J = αm - D∇m at each domain boundary
    to verify mass conservation and identify sources of mass loss.
    """

    def __init__(
        self,
        dimension: int,
        spacing: tuple[float, ...] | float,
        dt: float = 1.0,
        conservation_tolerance: float = 1e-10,
    ):
        """Initialize flux diagnostics.

        Args:
            dimension: Problem dimension (1, 2, or 3)
            spacing: Grid spacing (dx,) for 1D, (dx, dy) for 2D, etc.
            dt: Time step size (for integrating boundary flux over time)
            conservation_tolerance: Threshold for declaring flux "conserved"
        """
        self.dimension = dimension
        self.spacing = (spacing,) if isinstance(spacing, (int, float)) else tuple(spacing)
        self.dt = dt
        self.tolerance = conservation_tolerance

        # Flux history
        self._flux_history: list[BoundaryFluxResult] = []
        self._initial_mass: float | None = None

        # Boundary names by dimension
        self._boundary_names = self._get_boundary_names(dimension)

    @staticmethod
    def _get_boundary_names(dimension: int) -> list[str]:
        """Get boundary names for given dimension."""
        if dimension == 1:
            return ["left", "right"]
        elif dimension == 2:
            return ["left", "right", "bottom", "top"]
        elif dimension == 3:
            return ["left", "right", "bottom", "top", "back", "front"]
        else:
            raise ValueError(f"Unsupported dimension: {dimension}")

    def compute_boundary_flux(
        self,
        m: NDArray,
        velocity_field: NDArray | None = None,
        u_field: NDArray | None = None,
        coupling_coefficient: float = 1.0,
        diffusion: float | NDArray = 0.0,
    ) -> BoundaryFluxResult:
        """Compute boundary flux for current density field.

        The flux J = αm - D∇m is computed at each boundary.
        Positive flux means mass is leaving through that boundary.

        Args:
            m: Density field, shape (Nx+1,) for 1D, (Nx+1, Ny+1) for 2D
            velocity_field: Precomputed velocity α, same shape as m
                If None, computed from u_field as α = -coupling * ∇u
            u_field: Value function u(x), used to compute velocity if velocity_field is None
            coupling_coefficient: Coefficient λ in α = -∇u/λ
            diffusion: Diffusion coefficient D = σ²/2, scalar or spatially varying

        Returns:
            BoundaryFluxResult with flux breakdown
        """
        # Compute velocity field if not provided
        if velocity_field is None:
            if u_field is not None:
                velocity_field = self._compute_velocity_from_u(u_field, coupling_coefficient)
            else:
                # Zero velocity
                velocity_field = np.zeros_like(m)

        # Compute fluxes based on dimension
        if self.dimension == 1:
            result = self._compute_flux_1d(m, velocity_field, diffusion)
        elif self.dimension == 2:
            result = self._compute_flux_2d(m, velocity_field, diffusion)
        else:
            raise NotImplementedError(f"Flux diagnostics not implemented for {self.dimension}D")

        # Track history
        self._flux_history.append(result)
        if self._initial_mass is None:
            self._initial_mass = result.total_mass

        return result

    def _compute_velocity_from_u(self, u: NDArray, coupling: float) -> NDArray:
        """Compute velocity field α = -∇u / coupling from value function."""
        if self.dimension == 1:
            dx = self.spacing[0]
            # Central difference for interior, one-sided at boundaries
            alpha = np.zeros_like(u)
            alpha[1:-1] = -(u[2:] - u[:-2]) / (2 * dx * coupling)
            alpha[0] = -(u[1] - u[0]) / (dx * coupling)
            alpha[-1] = -(u[-1] - u[-2]) / (dx * coupling)
            return alpha
        elif self.dimension == 2:
            dx, dy = self.spacing
            alpha_x = np.zeros_like(u)
            alpha_y = np.zeros_like(u)

            # Interior: central differences
            alpha_x[1:-1, :] = -(u[2:, :] - u[:-2, :]) / (2 * dx * coupling)
            alpha_y[:, 1:-1] = -(u[:, 2:] - u[:, :-2]) / (2 * dy * coupling)

            # Boundaries: one-sided
            alpha_x[0, :] = -(u[1, :] - u[0, :]) / (dx * coupling)
            alpha_x[-1, :] = -(u[-1, :] - u[-2, :]) / (dx * coupling)
            alpha_y[:, 0] = -(u[:, 1] - u[:, 0]) / (dy * coupling)
            alpha_y[:, -1] = -(u[:, -1] - u[:, -2]) / (dy * coupling)

            # Return as tuple (will need adjustment for interface)
            return np.stack([alpha_x, alpha_y], axis=-1)
        else:
            raise NotImplementedError

    def _compute_flux_1d(
        self,
        m: NDArray,
        alpha: NDArray,
        D: float | NDArray,
    ) -> BoundaryFluxResult:
        """Compute 1D boundary flux.

        J = αm - D(∂m/∂x)

        At left boundary (x=0): J_left, positive = rightward
        At right boundary (x=L): J_right, positive = rightward
        Net flux out = J_right - J_left
        """
        dx = self.spacing[0]

        # Total mass
        total_mass = float(np.sum(m) * dx)

        # Diffusion coefficient handling
        D_left = D if np.isscalar(D) else D[0]
        D_right = D if np.isscalar(D) else D[-1]

        # Left boundary flux (positive = into domain from left = entering)
        # J_left = α[0]*m[0] - D*∂m/∂x|_0
        # Using one-sided difference: ∂m/∂x|_0 ≈ (m[1] - m[0]) / dx
        dm_dx_left = (m[1] - m[0]) / dx
        advection_left = alpha[0] * m[0]
        diffusion_left = -D_left * dm_dx_left
        J_left = advection_left + diffusion_left

        # Right boundary flux (positive = leaving domain)
        # J_right = α[-1]*m[-1] - D*∂m/∂x|_L
        dm_dx_right = (m[-1] - m[-2]) / dx
        advection_right = alpha[-1] * m[-1]
        diffusion_right = -D_right * dm_dx_right
        J_right = advection_right + diffusion_right

        # Net flux out of domain = flux leaving right - flux entering left
        # Convention: positive = mass leaving
        net_flux = J_right - J_left
        net_advection = advection_right - advection_left
        net_diffusion = diffusion_right - diffusion_left

        boundary_fluxes = {
            "left": float(J_left),
            "right": float(J_right),
        }

        is_conserved = abs(net_flux) < self.tolerance

        return BoundaryFluxResult(
            total_mass=total_mass,
            boundary_fluxes=boundary_fluxes,
            net_flux=float(net_flux),
            advection_flux=float(net_advection),
            diffusion_flux=float(net_diffusion),
            is_conserved=is_conserved,
            tolerance=self.tolerance,
        )

    def _compute_flux_2d(
        self,
        m: NDArray,
        alpha: NDArray,
        D: float | NDArray,
    ) -> BoundaryFluxResult:
        """Compute 2D boundary flux.

        J = α·m - D∇m

        Integrate flux along each boundary edge.
        """
        dx, dy = self.spacing

        # Total mass
        total_mass = float(np.sum(m) * dx * dy)

        # Handle velocity field shape
        if alpha.ndim == 2:
            # Scalar velocity (isotropic)
            alpha_x = alpha
            alpha_y = alpha
        else:
            # Vector velocity (Nx, Ny, 2)
            alpha_x = alpha[..., 0]
            alpha_y = alpha[..., 1]

        # Diffusion coefficient
        D_val = D if np.isscalar(D) else np.mean(D)

        boundary_fluxes = {}
        total_advection = 0.0
        total_diffusion = 0.0

        # Left boundary (x=0): integrate over y, outward normal = (-1, 0)
        dm_dx_left = (m[1, :] - m[0, :]) / dx
        adv_left = -alpha_x[0, :] * m[0, :]  # -α_x because normal is -x
        diff_left = D_val * dm_dx_left  # D*∂m/∂x, outward is -x so sign flips
        J_left = float(np.sum(adv_left + diff_left) * dy)
        boundary_fluxes["left"] = J_left
        total_advection += float(np.sum(adv_left) * dy)
        total_diffusion += float(np.sum(diff_left) * dy)

        # Right boundary (x=L): outward normal = (+1, 0)
        dm_dx_right = (m[-1, :] - m[-2, :]) / dx
        adv_right = alpha_x[-1, :] * m[-1, :]
        diff_right = -D_val * dm_dx_right
        J_right = float(np.sum(adv_right + diff_right) * dy)
        boundary_fluxes["right"] = J_right
        total_advection += float(np.sum(adv_right) * dy)
        total_diffusion += float(np.sum(diff_right) * dy)

        # Bottom boundary (y=0): outward normal = (0, -1)
        dm_dy_bottom = (m[:, 1] - m[:, 0]) / dy
        adv_bottom = -alpha_y[:, 0] * m[:, 0]
        diff_bottom = D_val * dm_dy_bottom
        J_bottom = float(np.sum(adv_bottom + diff_bottom) * dx)
        boundary_fluxes["bottom"] = J_bottom
        total_advection += float(np.sum(adv_bottom) * dx)
        total_diffusion += float(np.sum(diff_bottom) * dx)

        # Top boundary (y=L): outward normal = (0, +1)
        dm_dy_top = (m[:, -1] - m[:, -2]) / dy
        adv_top = alpha_y[:, -1] * m[:, -1]
        diff_top = -D_val * dm_dy_top
        J_top = float(np.sum(adv_top + diff_top) * dx)
        boundary_fluxes["top"] = J_top
        total_advection += float(np.sum(adv_top) * dx)
        total_diffusion += float(np.sum(diff_top) * dx)

        # Net flux (sum of all boundary fluxes, all pointing outward)
        net_flux = sum(boundary_fluxes.values())
        is_conserved = abs(net_flux) < self.tolerance

        return BoundaryFluxResult(
            total_mass=total_mass,
            boundary_fluxes=boundary_fluxes,
            net_flux=float(net_flux),
            advection_flux=float(total_advection),
            diffusion_flux=float(total_diffusion),
            is_conserved=is_conserved,
            tolerance=self.tolerance,
        )

    def get_summary(self) -> FluxSummary:
        """Get summary of flux diagnostics over all tracked timesteps.

        Computes both total mass drift and the breakdown into:
        - Physical boundary flux (expected mass loss through boundaries)
        - Algorithmic leak (discretization error)

        For conservative Flux FDM: algorithmic_leak should be ~0
        For Gradient FDM: algorithmic_leak may be significant

        Returns:
            FluxSummary with mass conservation statistics
        """
        if not self._flux_history:
            raise ValueError("No flux data recorded. Call compute_boundary_flux first.")

        initial_mass = self._flux_history[0].total_mass
        final_mass = self._flux_history[-1].total_mass
        actual_mass_change = final_mass - initial_mass  # Signed change
        mass_drift = abs(actual_mass_change)
        mass_drift_pct = mass_drift / initial_mass * 100 if initial_mass > 0 else 0.0

        net_fluxes = [abs(f.net_flux) for f in self._flux_history]
        max_net = max(net_fluxes)
        mean_net = sum(net_fluxes) / len(net_fluxes)

        # Per-boundary breakdown
        boundary_breakdown = {}
        for name in self._boundary_names:
            fluxes = [f.boundary_fluxes.get(name, 0.0) for f in self._flux_history]
            boundary_breakdown[name] = {
                "mean": float(np.mean(fluxes)),
                "max": float(np.max(np.abs(fluxes))),
                "total": float(np.sum(fluxes)),
            }

        # ============================================================
        # KEY: Separate physical flux from algorithmic leak
        # ============================================================
        # Integrate net boundary flux over time: expected_mass_change = -dt * sum(net_flux)
        # (negative because positive net_flux means mass leaving -> mass decreasing)
        signed_net_fluxes = [f.net_flux for f in self._flux_history]
        integrated_boundary_flux = -self.dt * sum(signed_net_fluxes)

        # Algorithmic leak = what the actual change is beyond what boundary flux explains
        # For conservative methods: this should be ~0
        # For non-conservative methods: this captures discretization error
        algorithmic_leak = actual_mass_change - integrated_boundary_flux
        algorithmic_leak_pct = abs(algorithmic_leak) / initial_mass * 100 if initial_mass > 0 else 0.0

        # Conservative if mass drift is small
        is_conservative = mass_drift_pct < 1.0  # 1% threshold

        return FluxSummary(
            initial_mass=initial_mass,
            final_mass=final_mass,
            mass_drift_absolute=mass_drift,
            mass_drift_percent=mass_drift_pct,
            max_net_flux=max_net,
            mean_net_flux=mean_net,
            flux_history=self._flux_history,
            is_conservative=is_conservative,
            boundary_breakdown=boundary_breakdown,
            integrated_boundary_flux=integrated_boundary_flux,
            algorithmic_leak=algorithmic_leak,
            algorithmic_leak_percent=algorithmic_leak_pct,
        )

    def reset(self) -> None:
        """Clear flux history for new simulation."""
        self._flux_history = []
        self._initial_mass = None

    def print_summary(self) -> None:
        """Print human-readable flux diagnostics summary."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("FLUX DIAGNOSTICS SUMMARY")
        print("=" * 60)

        status = "CONSERVED" if summary.is_conservative else "NOT CONSERVED"
        print(f"Status: {status}")
        print(f"Timesteps analyzed: {len(summary.flux_history)}")
        print()

        print("Mass Balance:")
        print(f"  Initial mass: {summary.initial_mass:.6f}")
        print(f"  Final mass:   {summary.final_mass:.6f}")
        print(f"  Drift:        {summary.mass_drift_absolute:.2e} ({summary.mass_drift_percent:.2f}%)")
        print()

        print("Mass Change Breakdown (KEY DIAGNOSTIC):")
        print(f"  Physical boundary flux:  {summary.integrated_boundary_flux:+.2e}")
        print(f"  Algorithmic leak:        {summary.algorithmic_leak:+.2e} ({summary.algorithmic_leak_percent:.2f}%)")
        if summary.algorithmic_leak_percent < 0.01:
            print("  --> Conservative discretization confirmed")
        else:
            print("  --> Non-conservative discretization detected")
        print()

        print("Net Flux Statistics:")
        print(f"  Max |net flux|:  {summary.max_net_flux:.2e}")
        print(f"  Mean |net flux|: {summary.mean_net_flux:.2e}")
        print()

        print("Per-Boundary Breakdown:")
        for name, stats in summary.boundary_breakdown.items():
            print(f"  {name:8s}: mean={stats['mean']:+.2e}, max={stats['max']:.2e}")

        print("=" * 60 + "\n")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def compute_mass_conservation_error(
    M: NDArray,
    spacing: tuple[float, ...] | float,
) -> dict[str, float]:
    """Quick mass conservation check without full flux diagnostics.

    Args:
        M: Density evolution array, shape (Nt+1, Nx+1) for 1D
           or (Nt+1, Nx+1, Ny+1) for 2D
        spacing: Grid spacing

    Returns:
        Dictionary with mass conservation statistics
    """
    if isinstance(spacing, (int, float)):
        spacing = (spacing,)

    # Compute cell volume
    cell_volume = float(np.prod(spacing))

    # Compute mass at each timestep
    if M.ndim == 2:
        # 1D: (Nt+1, Nx+1)
        mass_history = np.sum(M, axis=1) * cell_volume
    elif M.ndim == 3:
        # 2D: (Nt+1, Nx+1, Ny+1)
        mass_history = np.sum(M, axis=(1, 2)) * cell_volume
    else:
        raise ValueError(f"Unsupported array dimension: {M.ndim}")

    initial_mass = mass_history[0]
    final_mass = mass_history[-1]
    drift = abs(final_mass - initial_mass)
    drift_pct = drift / initial_mass * 100 if initial_mass > 0 else 0.0

    return {
        "initial_mass": float(initial_mass),
        "final_mass": float(final_mass),
        "mass_drift_absolute": float(drift),
        "mass_drift_percent": float(drift_pct),
        "mass_history": mass_history.tolist(),
        "is_conservative": drift_pct < 1.0,
    }


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    """Quick smoke test for flux diagnostics."""
    print("Testing FluxDiagnostics...")

    # Test 1D
    print("\n1. Testing 1D flux computation...")
    Nx = 50
    dx = 1.0 / Nx
    x = np.linspace(0, 1, Nx + 1)

    # Gaussian density
    m = np.exp(-((x - 0.5) ** 2) / 0.02)
    m /= np.sum(m) * dx

    # Velocity field (rightward)
    alpha = np.ones_like(m) * 0.5

    # Create diagnostics
    diag_1d = FluxDiagnostics(dimension=1, spacing=dx)
    result = diag_1d.compute_boundary_flux(m, velocity_field=alpha, diffusion=0.01)

    print(f"   Total mass: {result.total_mass:.6f}")
    print(f"   Left flux:  {result.boundary_fluxes['left']:+.4e}")
    print(f"   Right flux: {result.boundary_fluxes['right']:+.4e}")
    print(f"   Net flux:   {result.net_flux:+.4e}")
    print(f"   Conserved:  {result.is_conserved}")

    # Test 2D
    print("\n2. Testing 2D flux computation...")
    Nx, Ny = 20, 20
    dx, dy = 1.0 / Nx, 1.0 / Ny
    x = np.linspace(0, 1, Nx + 1)
    y = np.linspace(0, 1, Ny + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Gaussian density
    m_2d = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)
    m_2d /= np.sum(m_2d) * dx * dy

    # Zero velocity (pure diffusion)
    alpha_2d = np.zeros((Nx + 1, Ny + 1))

    diag_2d = FluxDiagnostics(dimension=2, spacing=(dx, dy))
    result_2d = diag_2d.compute_boundary_flux(m_2d, velocity_field=alpha_2d, diffusion=0.01)

    print(f"   Total mass: {result_2d.total_mass:.6f}")
    for name, flux in result_2d.boundary_fluxes.items():
        print(f"   {name:8s}: {flux:+.4e}")
    print(f"   Net flux:   {result_2d.net_flux:+.4e}")

    # Test mass conservation helper
    print("\n3. Testing mass conservation helper...")
    Nt = 10
    M_evolution = np.zeros((Nt + 1, Nx + 1))
    for t in range(Nt + 1):
        # Simulate spreading Gaussian
        width = 0.02 + 0.01 * t / Nt
        M_evolution[t] = np.exp(-((x - 0.5) ** 2) / width)
        M_evolution[t] /= np.sum(M_evolution[t]) * dx

    stats = compute_mass_conservation_error(M_evolution, spacing=dx)
    print(f"   Initial mass: {stats['initial_mass']:.6f}")
    print(f"   Final mass:   {stats['final_mass']:.6f}")
    print(f"   Drift:        {stats['mass_drift_percent']:.2f}%")
    print(f"   Conservative: {stats['is_conservative']}")

    print("\nAll smoke tests passed!")
