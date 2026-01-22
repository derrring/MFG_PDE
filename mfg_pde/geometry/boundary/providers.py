"""
Dynamic boundary condition value providers (Issue #625).

This module implements the callback-provider pattern for state-dependent
boundary conditions. Instead of baking coupling logic into solvers, the
BC *intent* is stored in the BCSegment and resolved to concrete values
by the coupling iterator.

Architecture:
------------
1. BCValueProvider protocol defines the compute(state) -> value contract
2. Concrete providers (e.g., AdjointConsistentProvider) implement the formula
3. BCSegment.value can hold a provider (intent) or static value
4. FixedPointIterator resolves providers before passing BC to solvers

Benefits:
---------
- Solvers remain generic (no MFG coupling knowledge)
- New dynamic BCs = new provider class (no solver modification)
- Intent is explicit in BC object, not hidden in string flags
- Providers are unit-testable in isolation

Example:
--------
    >>> from mfg_pde.geometry.boundary.providers import AdjointConsistentProvider
    >>> from mfg_pde.geometry.boundary import BCSegment, BCType
    >>>
    >>> # Store intent in BCSegment
    >>> segment = BCSegment(
    ...     name="left_ac",
    ...     bc_type=BCType.ROBIN,
    ...     alpha=0.0, beta=1.0,
    ...     value=AdjointConsistentProvider(side="left", sigma=0.2),
    ...     boundary="x_min",
    ... )
    >>>
    >>> # Later, iterator resolves provider with current state
    >>> state = {'m_current': m, 'geometry': geometry, 'sigma': 0.2}
    >>> concrete_value = segment.value.compute(state)

References:
-----------
- Issue #625: Dynamic BC value provider architecture
- Issue #574: Original adjoint-consistent BC implementation
- docs/development/boundary_condition_handling_summary.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Protocol Definition
# =============================================================================


@runtime_checkable
class BCValueProvider(Protocol):
    """
    Protocol for dynamic boundary condition value generation.

    Providers compute BC values from the current system state during
    iteration. This enables state-dependent BCs (like adjoint-consistent
    reflecting boundaries) without coupling logic in solvers.

    The protocol is runtime-checkable, allowing isinstance() checks:
        >>> if isinstance(segment.value, BCValueProvider):
        ...     resolved = segment.value.compute(state)

    Methods:
        compute: Generate BC value(s) from current iteration state.

    Note:
        Providers should be lightweight and stateless where possible.
        Heavy computation should be cached or precomputed.
    """

    def compute(self, state: dict[str, Any]) -> float | NDArray[np.floating]:
        """
        Compute BC value(s) from current system state.

        Args:
            state: Dictionary containing iteration state. Standard keys:
                - 'm_current': Current FP density array
                - 'U_current': Current value function array
                - 'geometry': Problem geometry object
                - 'sigma': Diffusion coefficient
                - 't': Current time (for time-dependent problems)
                - 'iteration': Current Picard iteration number

        Returns:
            BC value as scalar (single boundary point) or array
            (multiple boundary points).

        Raises:
            KeyError: If required state keys are missing.
            ValueError: If state values have unexpected shapes/types.
        """
        ...


# =============================================================================
# Abstract Base Class (for inheritance-based implementations)
# =============================================================================


class BaseBCValueProvider(ABC):
    """
    Abstract base class for BC value providers.

    Use this when you need shared infrastructure (e.g., caching, logging)
    across provider implementations. For simple providers, implementing
    the BCValueProvider protocol directly is sufficient.
    """

    @abstractmethod
    def compute(self, state: dict[str, Any]) -> float | NDArray[np.floating]:
        """Compute BC value(s) from state. See BCValueProvider.compute()."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# =============================================================================
# Concrete Providers
# =============================================================================


class AdjointConsistentProvider(BaseBCValueProvider):
    """
    Provider for adjoint-consistent Robin BC values (Issue #574).

    Computes the Robin BC value for reflecting boundaries that maintains
    adjoint consistency between HJB and FP equations:

        g = -sigma^2/2 * d(ln m)/dn

    where:
        - sigma: diffusion coefficient
        - m: current FP density
        - d/dn: outward normal derivative at boundary

    This is used for reflecting boundaries where the stall point lies
    at or near the domain boundary. The formula ensures the HJB and FP
    BCs are consistent at equilibrium.

    Attributes:
        side: Boundary side ("left" or "right" for 1D)
        sigma: Diffusion coefficient (can be None to read from state)
        regularization: Small constant to prevent log(0)

    Example:
        >>> provider = AdjointConsistentProvider(side="left", sigma=0.2)
        >>> state = {'m_current': m_array, 'geometry': geom}
        >>> g_left = provider.compute(state)  # Robin BC value at left boundary
    """

    def __init__(
        self,
        side: str,
        sigma: float | None = None,
        regularization: float = 1e-10,
    ) -> None:
        """
        Initialize adjoint-consistent BC provider.

        Args:
            side: Boundary side identifier ("left", "right" for 1D;
                  will extend to "x_min", "x_max", etc. for nD)
            sigma: Diffusion coefficient. If None, reads from state['sigma'].
            regularization: Small positive constant added to density
                           to prevent log(0). Default 1e-10.
        """
        if side not in ("left", "right", "x_min", "x_max"):
            raise ValueError(f"side must be 'left', 'right', 'x_min', or 'x_max', got '{side}'")
        self.side = side
        self.sigma = sigma
        self.regularization = regularization

    def compute(self, state: dict[str, Any]) -> float:
        """
        Compute adjoint-consistent Robin BC value.

        Args:
            state: Must contain:
                - 'm_current': FP density array (interior points)
                - 'geometry': Geometry object with get_grid_spacing()
                - 'sigma': Diffusion coefficient (if not set in __init__)

        Returns:
            Robin BC value: g = -sigma^2/2 * d(ln m)/dn

        Raises:
            KeyError: If required state keys missing
            ValueError: If density has invalid shape
        """
        # Get density
        m = state.get("m_current")
        if m is None:
            raise KeyError("AdjointConsistentProvider requires 'm_current' in state")

        # Handle time-dependent density (take final time slice)
        if m.ndim > 1:
            m = m[-1, :]  # Final time slice for backward HJB

        # Get geometry for grid spacing
        geometry = state.get("geometry")
        if geometry is None:
            raise KeyError("AdjointConsistentProvider requires 'geometry' in state")

        dx = geometry.get_grid_spacing()[0]

        # Get sigma
        sigma = self.sigma
        if sigma is None:
            sigma = state.get("sigma")
            if sigma is None:
                raise KeyError("AdjointConsistentProvider: sigma not set and not in state")

        # Compute log-density gradient at boundary
        from mfg_pde.geometry.boundary.bc_coupling import (
            compute_boundary_log_density_gradient_1d,
        )

        # Normalize side name
        side = self.side
        if side == "x_min":
            side = "left"
        elif side == "x_max":
            side = "right"

        grad_ln_m = compute_boundary_log_density_gradient_1d(m, dx, side, self.regularization)

        # Robin BC value: g = -sigma^2/2 * d(ln m)/dn
        return -(sigma**2) / 2 * grad_ln_m

    def __repr__(self) -> str:
        sigma_str = f"{self.sigma}" if self.sigma is not None else "from_state"
        return f"AdjointConsistentProvider(side='{self.side}', sigma={sigma_str})"


class ConstantProvider(BaseBCValueProvider):
    """
    Trivial provider that returns a constant value.

    Useful for testing and as a reference implementation.
    In practice, use a float directly in BCSegment.value instead.
    """

    def __init__(self, value: float) -> None:
        self.value = value

    def compute(self, state: dict[str, Any]) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"ConstantProvider({self.value})"


# =============================================================================
# Utility Functions
# =============================================================================


def is_provider(value: Any) -> bool:
    """
    Check if a value is a BC value provider.

    Args:
        value: The value to check (from BCSegment.value)

    Returns:
        True if value implements BCValueProvider protocol

    Example:
        >>> from mfg_pde.geometry.boundary.providers import is_provider
        >>> is_provider(0.0)  # False
        >>> is_provider(AdjointConsistentProvider("left", 0.2))  # True
    """
    return isinstance(value, BCValueProvider)


def resolve_provider(
    value: float | BCValueProvider,
    state: dict[str, Any],
) -> float:
    """
    Resolve a BC value, computing if it's a provider.

    Args:
        value: Static value or provider
        state: Current iteration state (passed to provider.compute())

    Returns:
        Resolved float value

    Example:
        >>> value = AdjointConsistentProvider("left", 0.2)
        >>> resolved = resolve_provider(value, state)  # Calls compute()
        >>> resolve_provider(1.5, state)  # Returns 1.5 unchanged
    """
    if is_provider(value):
        result = value.compute(state)
        return float(result) if not isinstance(result, np.ndarray) else result
    return float(value)


# =============================================================================
# Smoke Test
# =============================================================================

if __name__ == "__main__":
    """Quick validation of provider architecture."""
    print("Testing BC value providers...")
    print()

    # Test 1: Protocol check
    print("Test 1: Protocol compliance")
    provider = AdjointConsistentProvider(side="left", sigma=0.2)
    assert isinstance(provider, BCValueProvider), "Should implement protocol"
    assert is_provider(provider), "is_provider() should return True"
    assert not is_provider(1.5), "Float should not be a provider"
    print("  Protocol checks passed")
    print()

    # Test 2: Compute with mock state
    print("Test 2: Compute with mock state")

    # Create mock geometry
    class MockGeometry:
        def get_grid_spacing(self):
            return [0.1]

    # Exponential density: m(x) = exp(-x), so d(ln m)/dx = -1
    x = np.linspace(0, 1, 11)
    m = np.exp(-x)

    state = {
        "m_current": m,
        "geometry": MockGeometry(),
        "sigma": 0.2,
    }

    # Left boundary: outward normal is -x, so d(ln m)/dn = -(-1) = 1
    # g = -0.2^2/2 * 1 = -0.02
    left_provider = AdjointConsistentProvider(side="left", sigma=0.2)
    g_left = left_provider.compute(state)
    expected_left = -(0.2**2) / 2 * 1.0  # -0.02
    print(f"  Left BC value: {g_left:.6f} (expected ~ {expected_left:.6f})")

    # Right boundary: outward normal is +x, so d(ln m)/dn = -1
    # g = -0.2^2/2 * (-1) = 0.02
    right_provider = AdjointConsistentProvider(side="right", sigma=0.2)
    g_right = right_provider.compute(state)
    expected_right = -(0.2**2) / 2 * (-1.0)  # 0.02
    print(f"  Right BC value: {g_right:.6f} (expected ~ {expected_right:.6f})")
    print()

    # Test 3: resolve_provider utility
    print("Test 3: resolve_provider utility")
    resolved = resolve_provider(left_provider, state)
    assert abs(resolved - g_left) < 1e-10, "Should match direct compute"
    resolved_static = resolve_provider(42.0, state)
    assert resolved_static == 42.0, "Static value should pass through"
    print("  resolve_provider works correctly")
    print()

    # Test 4: Sigma from state
    print("Test 4: Sigma from state (not constructor)")
    provider_no_sigma = AdjointConsistentProvider(side="left", sigma=None)
    g_from_state = provider_no_sigma.compute(state)
    assert abs(g_from_state - g_left) < 1e-10, "Should use sigma from state"
    print("  Sigma correctly read from state when not in constructor")
    print()

    print("All provider tests passed!")
