# Phase 2.5: Anisotropic Diffusion Tensors

**Status**: Planning | **Priority**: Medium | **Effort**: 3-5 days
**Created**: 2025-11-19 | **Branch**: `feature/anisotropic-diffusion-tensors`

## Overview

Extend MFG_PDE to support anisotropic diffusion via diffusion tensors **D(t, x, m)**, enabling direction-dependent diffusion coefficients. This generalizes scalar diffusion σ²(t, x, m) to full tensors supporting cross-diffusion and preferential directions.

### Mathematical Formulation

#### Scalar Diffusion (Phase 2 - Current)
```
∂m/∂t = ∇ · (σ² ∇m) - ∇ · (α m)
```
where σ² is a scalar coefficient (possibly state-dependent).

#### Tensor Diffusion (Phase 2.5 - Target)
```
∂m/∂t = ∇ · (Σ ∇m) - ∇ · (α m)
```
where **Σ** is a d×d positive semi-definite diffusion tensor:
```
Σ = [σ₁₁  σ₁₂  ...  σ₁ₐ]
    [σ₂₁  σ₂₂  ...  σ₂ₐ]
    [... ...  ... ...]
    [σₐ₁  σₐ₂  ...  σₐₐ]
```

**Notation Conventions**:
- Scalar diffusion: σ² (lowercase sigma squared)
- Tensor diffusion: **Σ** (capital Sigma matrix)

**Three Interpretations of Diffusion**:

1. **Isotropic (scalar → tensor)**: σ² → **Σ = σ²I**
   - Same diffusion in all directions
   - Example: Σ = 0.1·I in 2D gives uniform diffusion

2. **Diagonal anisotropic (vector → tensor)**: σ = [σ₁, σ₂, ...] → **Σ = diag(σ₁², σ₂², ...)**
   - Different diffusion per direction, no cross-terms
   - Can also view as Σ = σσᵀ for diagonal σ
   - Example: σ = [0.2, 0.05] → Σ = diag([0.04, 0.0025])

3. **General anisotropic (full tensor)**: **Σ** directly specified
   - Arbitrary symmetric positive semi-definite matrix
   - Includes cross-diffusion terms σᵢⱼ
   - Example: Σ = [[0.1, 0.02], [0.02, 0.05]]

**Connection to SDEs**:
If the underlying stochastic process is dX = α dt + σ̃ dW where σ̃ is a d×d diffusion matrix,
then the Fokker-Planck diffusion tensor is **Σ = (σ̃σ̃ᵀ)/2**.

In particular:
- If σ̃ is diagonal, then Σ = (σ̃σ̃ᵀ)/2 is also diagonal
- If σ̃ is a scaled identity σ̃ = σI, then Σ = σ²I/2

However, in our PDE-based framework, users specify **Σ** directly without referencing σ̃.

**Key Properties**:
- **Symmetry**: Σ = Σᵀ (for physical realizability)
- **Positive semi-definite**: xᵀΣx ≥ 0 for all x (ensures well-posedness)
- **State-dependent**: Σ = Σ(t, x, m) (supports callable tensors)

## Motivation

### Physical Applications

1. **Anisotropic Crowd Dynamics**
   - Pedestrians move more easily along corridors than perpendicular
   - σ₁₁ > σ₂₂ in corridor direction

2. **Traffic Flow with Lane Structure**
   - Cars move faster along lanes than across lanes
   - Cross-diffusion σ₁₂ captures lane-changing behavior

3. **Environmental Heterogeneity**
   - Terrain with ridges (preferential movement along ridges)
   - Porous media with directional permeability

4. **Multi-Population Segregation**
   - Cross-diffusion terms σ₁₂ model inter-population interactions
   - Competitive or cooperative dynamics

### Example: Pedestrian Corridor

```python
def corridor_diffusion(t, x, m):
    """
    Anisotropic diffusion in a corridor.

    Assume x[0] = longitudinal (along corridor)
           x[1] = lateral (across corridor)

    Returns Σ matrix (capital Sigma, not to be confused with scalar σ²).
    """
    sigma_parallel = 0.2   # High diffusion along corridor
    sigma_perpendicular = 0.05  # Low diffusion across corridor

    return np.array([
        [sigma_parallel, 0.0],
        [0.0, sigma_perpendicular]
    ])
```

## Design Decisions

### 1. Tensor Representation

**Options**:
- (A) Full tensor: d×d array
- (B) Symmetric tensor: Store only upper/lower triangle
- (C) Diagonal tensor: Store only diagonal entries
- (D) Eigenvalue decomposition: Store eigenvalues + rotation

**Decision**: Use **(A) Full tensor** initially, optimize later if needed.

**Rationale**:
- Simple to implement and understand
- Allows arbitrary positive semi-definite tensors
- Memory overhead is O(d²) which is acceptable for d ≤ 10
- Can add optimized paths for diagonal/symmetric cases later

### 2. Callable Tensor API

**Signature**:
```python
def diffusion_tensor(t: float, x: NDArray, m: NDArray) -> NDArray:
    """
    Compute diffusion tensor Σ at given state.

    Args:
        t: Current time (scalar)
        x: Spatial coordinates (shape depends on dimension)
           - 1D: (Nx,)
           - 2D: (Nx, Ny) or tuple of 1D arrays
           - nD: tuple of 1D arrays per dimension
        m: Current density (same shape as spatial grid)

    Returns:
        Diffusion tensor Σ (capital Sigma) with shape:
        - Scalar: float (isotropic, Σ = σ²I)
        - 1D: (Nx,) array (spatially varying scalar)
        - 2D: (2, 2) or (Nx, Ny, 2, 2) tensor
        - nD: (d, d) or (Nx₁, ..., Nxₐ, d, d) tensor
    """
    # Return tensor at each grid point
```

**Key Design Choices**:
- Callable returns either constant tensor (d×d) or spatially varying (N₁×...×Nₐ×d×d)
- Backward compatible: scalar return interpreted as isotropic D = σ²I

### 3. Type Protocol Extension

Update `mfg_pde/types/pde_coefficients.py`:

```python
from typing import Protocol, Union
import numpy as np
from numpy.typing import NDArray

class DiffusionCallable(Protocol):
    """Protocol for callable diffusion coefficients."""

    def __call__(
        self,
        t: float,
        x: Union[NDArray[np.floating], tuple[NDArray[np.floating], ...]],
        m: NDArray[np.floating]
    ) -> Union[float, NDArray[np.floating]]:
        """
        Evaluate diffusion at given state.

        Returns:
            - float: Isotropic scalar diffusion
            - NDArray (Nx,): Spatially varying scalar
            - NDArray (d, d): Constant tensor
            - NDArray (..., d, d): Spatially varying tensor
        """
        ...

# Type alias for all diffusion representations
DiffusionField = Union[
    None,  # Use problem.sigma default
    float,  # Constant scalar
    int,  # Constant scalar (converted to float)
    NDArray[np.floating],  # Array or tensor
    DiffusionCallable  # Callable returning scalar or tensor
]
```

## Implementation Plan

### Task 1: Extend Finite Difference Operators

**File**: `mfg_pde/alg/numerical/operators.py` (or create new `tensor_operators.py`)

**New Functions**:

```python
def divergence_tensor_diffusion_2d(
    m: NDArray,
    sigma_tensor: NDArray,  # Shape: (2, 2) or (Nx, Ny, 2, 2)
    dx: float,
    dy: float,
    boundary_conditions: BoundaryConditions
) -> NDArray:
    """
    Compute ∇ · (Σ ∇m) in 2D with tensor diffusion.

    Args:
        m: Density field (Nx, Ny)
        sigma_tensor: Diffusion tensor Σ
            - Constant: (2, 2) array
            - Spatially varying: (Nx, Ny, 2, 2) array
        dx, dy: Grid spacing
        boundary_conditions: BC specification

    Discretization:
        ∇ · (Σ ∇m) = ∂/∂x(σ₁₁ ∂m/∂x + σ₁₂ ∂m/∂y)
                    + ∂/∂y(σ₂₁ ∂m/∂x + σ₂₂ ∂m/∂y)

    Uses central differences with ghost cells for boundary conditions.
    """
    # Compute gradients: ∇m = (∂m/∂x, ∂m/∂y)
    # Compute flux: F = Σ ∇m = (σ₁₁ ∂m/∂x + σ₁₂ ∂m/∂y,
    #                            σ₂₁ ∂m/∂x + σ₂₂ ∂m/∂y)
    # Compute divergence: ∇ · F
    ...

def divergence_tensor_diffusion_nd(
    m: NDArray,
    sigma_tensor: NDArray,  # Shape: (d, d) or (N₁, ..., Nₐ, d, d)
    dx: tuple[float, ...],
    boundary_conditions: BoundaryConditions
) -> NDArray:
    """
    Compute ∇ · (Σ ∇m) in arbitrary dimensions.

    Args:
        sigma_tensor: Diffusion tensor Σ (capital Sigma)
            - Constant: (d, d) array
            - Spatially varying: (N₁, ..., Nₐ, d, d) array

    Generalization of 2D formula to d dimensions.
    """
    ...
```

**Challenges**:
- Handling spatially varying tensors (N×...×d×d indexing)
- Boundary conditions for cross-diffusion terms
- Efficient implementation (avoid explicit loops)

### Task 2: Update CoefficientField Abstraction

**File**: `mfg_pde/utils/pde_coefficients.py`

**Changes**:

```python
class CoefficientField:
    """Unified interface for scalar, array, tensor, and callable coefficients."""

    def __init__(
        self,
        field: DiffusionField,
        default_value: float,
        field_name: str = "coefficient",
        dimension: int = 1,
        is_tensor: bool = False  # NEW: Flag for tensor fields
    ):
        self.is_tensor = is_tensor
        # ... existing code ...

    def evaluate_at(
        self,
        timestep_idx: int,
        grid: Union[NDArray, tuple[NDArray, ...]],
        density: NDArray,
        dt: float | None = None
    ) -> Union[float, NDArray]:
        """
        Evaluate coefficient at specific timestep and state.

        Returns:
            - Scalar diffusion: float or (Nx,) array
            - Tensor diffusion: (d, d) or (..., d, d) array
        """
        if self.is_tensor:
            return self._evaluate_tensor(timestep_idx, grid, density, dt)
        else:
            # Existing scalar evaluation logic
            ...

    def _evaluate_tensor(self, timestep_idx, grid, density, dt):
        """Evaluate tensor diffusion field."""
        if self._is_callable:
            # Call user function
            D = self.field(t, grid, density)
            # Validate tensor shape and properties
            self._validate_tensor_output(D, density.shape)
            return D
        elif self._is_array:
            # Extract from precomputed tensor array
            return self._extract_tensor_from_array(timestep_idx, density.shape)
        else:
            # Scalar or constant tensor
            return self._scalar_to_isotropic_tensor(self.field, self.dimension)

    def _validate_tensor_output(self, sigma_tensor, grid_shape):
        """
        Validate tensor diffusion output.

        Args:
            sigma_tensor: Diffusion tensor Σ (capital Sigma)

        Checks:
        - Shape: (d, d) or (N₁, ..., Nₐ, d, d)
        - Symmetry: Σ = Σᵀ (within tolerance)
        - Positive semi-definite: eigenvalues ≥ 0
        - No NaN/Inf
        """
        ...

    def _scalar_to_isotropic_tensor(self, sigma_squared, d):
        """
        Convert scalar σ² to isotropic tensor Σ = σ²I.

        Args:
            sigma_squared: Scalar diffusion coefficient (σ²)
            d: Spatial dimension

        Returns:
            Σ = σ²I, a d×d isotropic diffusion tensor

        Note:
            In 1D: σ²I = σ² (scalar), which equals σσᵀ (1×1 outer product)
            In nD: σ²I = diag([σ², σ², ...]), isotropic diffusion

            This is equivalent to the SDE interpretation with σ̃ = σI:
                Σ = (σ̃σ̃ᵀ)/2 = (σI)(σI)ᵀ/2 = σ²I/2
            (We absorb the factor of 1/2 into the convention.)
        """
        return sigma_squared * np.eye(d)

    def _vector_to_diagonal_tensor(self, sigma_vector):
        """
        Convert vector σ to diagonal tensor Σ = diag(σ₁², σ₂², ...).

        Args:
            sigma_vector: 1D array of diffusion coefficients per dimension

        Returns:
            Σ = diag([σ₁², σ₂², ...]), a diagonal diffusion tensor

        Note:
            This can also be viewed as Σ = σσᵀ where σ is diagonal.
            Allows different diffusion per direction without cross-terms.
        """
        return np.diag(sigma_vector ** 2)
```

### Task 3: Update FP Solvers

**File**: `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py`

**Changes to `_solve_fp_nd_full_system()`**:

```python
def _solve_fp_nd_full_system(
    self,
    m0: NDArray,
    drift_field: DriftField | None = None,
    diffusion_field: DiffusionField | None = None,
    ...
) -> NDArray:
    """Solve FP equation in nD with tensor diffusion support."""

    # Detect if diffusion_field is tensor
    is_tensor_diffusion = self._check_if_tensor_diffusion(diffusion_field)

    # Create CoefficientField with tensor flag
    diffusion = CoefficientField(
        diffusion_field,
        problem.sigma,
        "diffusion_field",
        dimension=ndim,
        is_tensor=is_tensor_diffusion
    )

    for k in range(Nt):
        # Evaluate diffusion (scalar σ² or tensor Σ)
        sigma_at_k = diffusion.evaluate_at(k, grid.coordinates, M_current, dt)

        # Choose appropriate operator
        if isinstance(sigma_at_k, np.ndarray) and sigma_at_k.ndim > 1:
            # Tensor diffusion: Use tensor divergence operator
            diffusion_term = divergence_tensor_diffusion_nd(
                M_current, sigma_at_k, dx, boundary_conditions
            )
        else:
            # Scalar diffusion: Use existing scalar operator
            diffusion_term = laplacian_nd(M_current, sigma_at_k, dx, boundary_conditions)

        # ... rest of FP solver logic ...
```

### Task 4: Update HJB Solvers

**File**: `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py`

**Challenges**:
- HJB with tensor diffusion: H(x, p, Σ) = -½ pᵀ Σ p + ...
- Tensor Σ affects Hamiltonian evaluation
- May require iterative methods for nonlinear HJB

**Initial Approach**: Support tensor diffusion in FP only, keep HJB scalar.

**Rationale**:
- Most applications have anisotropic FP but isotropic HJB
- Simplifies implementation
- Can extend to HJB later if needed

### Task 5: Testing

**File**: `tests/unit/test_tensor_diffusion.py` (new)

**Test Cases**:

1. **Diagonal Tensor = Scalar**:
   ```python
   def test_diagonal_tensor_matches_scalar():
       """Σ = σ²I should match scalar diffusion σ²."""
       sigma_tensor = 0.1 * np.eye(2)  # Isotropic Σ
       # Compare FP solution with tensor Σ vs scalar σ² = 0.1
   ```

2. **Anisotropic 2D**:
   ```python
   def test_anisotropic_2d_diffusion():
       """Test Σ = diag(0.2, 0.05) in 2D."""
       # Higher diffusion in x than y
       sigma_tensor = np.diag([0.2, 0.05])
   ```

3. **Cross-Diffusion**:
   ```python
   def test_cross_diffusion_2d():
       """Test off-diagonal terms σ₁₂ ≠ 0."""
       sigma_tensor = np.array([[0.1, 0.02],
                                 [0.02, 0.1]])
       # Verify coupling between dimensions
   ```

4. **Callable Tensor**:
   ```python
   def test_callable_tensor_diffusion():
       """Test state-dependent tensor Σ(t, x, m)."""
       def density_dependent_anisotropy(t, x, m):
           # Higher anisotropy in high-density regions
           alpha = 0.5 * np.max(m)
           return np.array([[0.1 + alpha, 0.0],
                            [0.0, 0.05]])
   ```

5. **Validation**:
   ```python
   def test_tensor_validation():
       """Test validation of tensor properties."""
       # Non-symmetric tensor should raise error
       # Negative eigenvalue should raise error
       # NaN/Inf should raise error
   ```

### Task 6: Example

**File**: `examples/advanced/anisotropic_crowd_dynamics.py` (new)

**Scenario**: Pedestrian flow in a corridor with anisotropic diffusion.

```python
def corridor_tensor_diffusion(t, x, m):
    """
    Anisotropic diffusion tensor Σ for corridor flow.

    High diffusion along corridor (x), low across (y).
    Density-dependent: more anisotropic in high-density regions.

    Returns:
        Σ: Diffusion tensor (capital Sigma), shape (Nx, Ny, 2, 2)
    """
    # Grid coordinates
    X, Y = x  # Assume 2D grid

    # Base diffusion coefficients
    sigma_parallel = 0.2        # Along corridor
    sigma_perpendicular = 0.05  # Across corridor

    # Density-dependent enhancement
    m_max = np.max(m) if np.max(m) > 0 else 1.0
    anisotropy_factor = 1.0 + 2.0 * (m / m_max)  # More anisotropic when crowded

    # Construct tensor Σ at each grid point
    sigma_tensor = np.zeros((*m.shape, 2, 2))
    sigma_tensor[..., 0, 0] = sigma_parallel * anisotropy_factor
    sigma_tensor[..., 1, 1] = sigma_perpendicular / anisotropy_factor

    return sigma_tensor
```

## Success Metrics

### Phase 2.5 Goals

- [ ] Tensor diffusion operators implemented and tested
- [ ] CoefficientField extended to support tensors
- [ ] FP solvers accept tensor diffusion_field
- [ ] 90%+ test coverage for tensor features
- [ ] Example demonstrating anisotropic crowd dynamics
- [ ] Documentation and API reference complete

### Performance Targets

- Tensor diffusion overhead: <3x vs scalar (due to additional FD operations)
- Memory: O(d²) per grid point (acceptable for d ≤ 10)

## Implementation Timeline

**Day 1**: Planning and operator implementation
- Define tensor API and type protocols
- Implement `divergence_tensor_diffusion_2d()`
- Unit tests for 2D tensor operators

**Day 2**: CoefficientField extension
- Add `is_tensor` flag and tensor evaluation
- Implement tensor validation (symmetry, PSD)
- Unit tests for CoefficientField tensors

**Day 3**: FP solver integration
- Update `fp_fdm.py` to detect and handle tensors
- Route to tensor vs scalar operators
- Integration tests with known solutions

**Day 4**: nD generalization and HJB (optional)
- Implement `divergence_tensor_diffusion_nd()`
- Consider HJB tensor support (if time permits)

**Day 5**: Examples, documentation, benchmarking
- Create anisotropic crowd dynamics example
- Performance benchmarking vs scalar
- Update roadmap and API docs

## Open Questions

1. **HJB Tensor Support**: Should Phase 2.5 include HJB with tensor D, or FP only?
   - **Lean towards**: FP only initially, HJB in Phase 2.6 if needed

2. **Tensor Validation**: How strict should positive-definiteness checking be?
   - **Proposal**: Check eigenvalues ≥ -ε with ε = 1e-10 (numerical tolerance)

3. **Optimization**: Should we optimize for diagonal or symmetric tensors?
   - **Proposal**: Implement general case first, add optimized paths if profiling shows bottleneck

4. **Boundary Conditions**: How to handle tensor diffusion at boundaries?
   - **Proposal**: Apply component-wise (treat each flux component independently)

## References

- **Mathematics**: Evans PDE textbook (Chapter 7: Diffusion)
- **Numerics**: LeVeque FD methods (Chapter 9: Variable coefficients)
- **Applications**: Helbing crowd dynamics, cross-diffusion systems

---

**Next Action**: Implement tensor diffusion operators in 2D
**Dependencies**: Phase 2 complete ✅
**Estimated Completion**: 5 days after start
