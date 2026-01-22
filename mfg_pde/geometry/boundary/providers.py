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
    ...     value=AdjointConsistentProvider(side="left", diffusion=0.2),
    ...     boundary="x_min",
    ... )
    >>>
    >>> # Later, iterator resolves provider with current state
    >>> state = {'m_current': m, 'geometry': geometry, 'diffusion': 0.2}
    >>> concrete_value = segment.value.compute(state)

References:
-----------
- Issue #625: Dynamic BC value provider architecture
- Issue #574: Original adjoint-consistent BC implementation
- docs/development/boundary_condition_handling_summary.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

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
                - 'diffusion': Diffusion coefficient (canonical)
                - 'sigma': Diffusion coefficient (legacy alias, deprecated)
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

        g = -σ²/2 · d(ln m)/dn = -(diffusion²)/2 · d(ln m)/dn

    where:
        - diffusion: diffusion coefficient σ (NOT σ²)
        - m: current FP density
        - d/dn: outward normal derivative at boundary

    This is used for reflecting boundaries where the stall point lies
    at or near the domain boundary. The formula ensures the HJB and FP
    BCs are consistent at equilibrium.

    Attributes:
        side: Boundary side ("left" or "right" for 1D)
        diffusion: Diffusion coefficient (can be None to read from state)
        regularization: Small constant to prevent log(0)

    Example:
        >>> provider = AdjointConsistentProvider(side="left", diffusion=0.04)
        >>> state = {'m_current': m_array, 'geometry': geom}
        >>> g_left = provider.compute(state)  # Robin BC value at left boundary
    """

    # Side name aliases: map various conventions to canonical form
    # Canonical: "left"/"right" for 1D, "{axis}_min"/"{axis}_max" for nD
    _SIDE_ALIASES: ClassVar[dict[str, str]] = {
        # 1D aliases (all map to left/right)
        "left": "left",
        "right": "right",
        "x_min": "left",
        "x_max": "right",
        "min": "left",
        "max": "right",
        # 2D/3D aliases (for future nD support)
        "y_min": "y_min",
        "y_max": "y_max",
        "z_min": "z_min",
        "z_max": "z_max",
        "bottom": "y_min",
        "top": "y_max",
        "front": "z_min",
        "back": "z_max",
    }

    def __init__(
        self,
        side: str,
        diffusion: float | None = None,
        regularization: float = 1e-10,
        *,
        sigma: float | None = None,  # Legacy alias, deprecated
    ) -> None:
        """
        Initialize adjoint-consistent BC provider.

        Args:
            side: Boundary side identifier. Accepts multiple conventions:
                - 1D: "left", "right", "x_min", "x_max", "min", "max"
                - 2D: "y_min", "y_max", "bottom", "top"
                - 3D: "z_min", "z_max", "front", "back"
            diffusion: Diffusion coefficient σ. If None, reads from state.
            regularization: Small positive constant added to density
                           to prevent log(0). Default 1e-10.
            sigma: DEPRECATED. Use diffusion instead.
        """
        import warnings

        if side not in self._SIDE_ALIASES:
            valid = sorted(self._SIDE_ALIASES.keys())
            raise ValueError(f"side must be one of {valid}, got '{side}'")

        # Handle sigma as legacy alias (deprecated)
        if sigma is not None:
            warnings.warn(
                "Parameter 'sigma' is deprecated. Use 'diffusion' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if diffusion is None:
                diffusion = sigma

        # Store both original and normalized side names
        self._original_side = side
        self.side = self._SIDE_ALIASES[side]  # Normalize to canonical form
        self.diffusion = diffusion
        self.regularization = regularization

    def compute(self, state: dict[str, Any]) -> float:
        """
        Compute adjoint-consistent Robin BC value.

        Args:
            state: Must contain:
                - 'm_current': FP density array (interior points)
                - 'geometry': Geometry object with get_grid_spacing()
                - 'diffusion' or 'sigma': Diffusion coefficient σ
                  (if not set in __init__). 'diffusion' takes priority.

        Returns:
            Robin BC value: g = -σ²/2 * d(ln m)/dn

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

        # Get diffusion coefficient (try constructor, then state)
        diffusion = self.diffusion
        if diffusion is None:
            # Look for 'diffusion' first (canonical), then 'sigma' (legacy)
            diffusion = state.get("diffusion")
            if diffusion is None:
                diffusion = state.get("sigma")
            if diffusion is None:
                raise KeyError(
                    "AdjointConsistentProvider: diffusion not set in constructor "
                    "and neither 'diffusion' nor 'sigma' found in state"
                )

        # Compute log-density gradient at boundary
        # Delegate to dimension-aware function in bc_coupling module
        grad_ln_m = self._compute_boundary_log_gradient(m, geometry, self.side, self.regularization)

        # Robin BC value: g = -σ²/2 * d(ln m)/dn
        return -(diffusion**2) / 2 * grad_ln_m

    def _compute_boundary_log_gradient(
        self,
        m: NDArray[np.floating],
        geometry: Any,
        side: str,
        regularization: float,
    ) -> float:
        """
        Compute ∂ln(m)/∂n at boundary using geometry-appropriate method.

        This method dispatches to dimension-specific implementations.
        For nD support, geometry should provide gradient operators.

        Args:
            m: Density array (interior points)
            geometry: Geometry object with get_grid_spacing()
            side: Normalized boundary side identifier
            regularization: Small constant to prevent log(0)

        Returns:
            Outward normal derivative of ln(m) at boundary

        Note:
            Current implementation: 1D only (uses finite differences)
            Future (Issue #624): Use geometry.get_gradient_operator() for nD
        """
        dimension = getattr(geometry, "dimension", 1)

        if dimension == 1 or side in ("left", "right"):
            # 1D case: use finite difference implementation
            from mfg_pde.geometry.boundary.bc_coupling import (
                compute_boundary_log_density_gradient_1d,
            )

            dx = geometry.get_grid_spacing()[0]
            return compute_boundary_log_density_gradient_1d(m, dx, side, regularization)
        else:
            # nD case: requires geometry gradient operators (Issue #624)
            raise NotImplementedError(
                f"AdjointConsistentProvider: {dimension}D not yet implemented. "
                f"See Issue #624 for nD adjoint-consistent BC support."
            )

    def __repr__(self) -> str:
        diff_str = f"{self.diffusion}" if self.diffusion is not None else "from_state"
        # Show original side name for user clarity
        return f"AdjointConsistentProvider(side='{self._original_side}', diffusion={diff_str})"


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
    import warnings

    print("Testing BC value providers...")
    print()

    # Test 1: Protocol check
    print("Test 1: Protocol compliance")
    provider = AdjointConsistentProvider(side="left", diffusion=0.2)
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
        "diffusion": 0.2,  # Canonical parameter name
    }

    # Left boundary: outward normal is -x, so d(ln m)/dn = -(-1) = 1
    # g = -0.2^2/2 * 1 = -0.02
    left_provider = AdjointConsistentProvider(side="left", diffusion=0.2)
    g_left = left_provider.compute(state)
    expected_left = -(0.2**2) / 2 * 1.0  # -0.02
    print(f"  Left BC value: {g_left:.6f} (expected ~ {expected_left:.6f})")

    # Right boundary: outward normal is +x, so d(ln m)/dn = -1
    # g = -0.2^2/2 * (-1) = 0.02
    right_provider = AdjointConsistentProvider(side="right", diffusion=0.2)
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

    # Test 4: Diffusion from state (not constructor)
    print("Test 4: Diffusion from state (not constructor)")
    provider_no_diff = AdjointConsistentProvider(side="left", diffusion=None)
    g_from_state = provider_no_diff.compute(state)
    assert abs(g_from_state - g_left) < 1e-10, "Should use diffusion from state"
    print("  Diffusion correctly read from state when not in constructor")
    print()

    # Test 5: Legacy 'sigma' in state (backward compatibility)
    print("Test 5: Legacy 'sigma' in state (backward compatibility)")
    state_legacy = {
        "m_current": m,
        "geometry": MockGeometry(),
        "sigma": 0.2,  # Legacy parameter name
    }
    provider_legacy = AdjointConsistentProvider(side="left", diffusion=None)
    g_legacy = provider_legacy.compute(state_legacy)
    assert abs(g_legacy - g_left) < 1e-10, "Should accept 'sigma' in state for backward compat"
    print("  Legacy 'sigma' key in state works correctly")
    print()

    # Test 6: Deprecated 'sigma' parameter (backward compatibility)
    print("Test 6: Deprecated 'sigma' parameter (shows warning)")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        provider_deprecated = AdjointConsistentProvider(side="left", sigma=0.2)
        assert len(w) == 1, "Should emit deprecation warning"
        assert "deprecated" in str(w[0].message).lower()
        assert provider_deprecated.diffusion == 0.2, "sigma should map to diffusion"
    print("  Deprecated 'sigma' parameter works with warning")
    print()

    print("All provider tests passed!")
