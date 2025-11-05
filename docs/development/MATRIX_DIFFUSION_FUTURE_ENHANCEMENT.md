# Matrix Diffusion for Anisotropic MFG: Future Enhancement

**Created**: 2025-11-05
**Status**: Future Enhancement
**Related**: #245 (Dimension-agnostic MFG architecture)

## Current Implementation

The dimension-agnostic `MFGProblem` supports:
- **Scalar diffusion**: `diffusion_coeff: float` (constant σ)
- **Position-dependent scalar diffusion**: `diffusion_coeff: Callable` where `σ(x) → float`

This covers:
- Isotropic problems with constant noise
- Position-dependent noise intensity (e.g., varying terrain roughness)

## Existing Anisotropic Support

**Anisotropy is currently handled through the Hamiltonian**, not diffusion:

```python
def anisotropic_hamiltonian(x, m, p, t):
    """
    H = (1/2) p^T D^{-1} p + interaction(m)

    where D is the diffusion matrix for directional preferences
    """
    D_inv = get_anisotropy_matrix_inverse(x)
    p_arr = np.array(p)
    return 0.5 * p_arr @ D_inv @ p_arr + nu * m

problem = MFGProblem(
    ...,
    diffusion_coeff=0.02,  # Scalar σ for FP equation
    hamiltonian_func=anisotropic_hamiltonian  # Anisotropy encoded here
)
```

This works for problems where:
- **Kinetic energy** is anisotropic (different costs for different directions)
- **Diffusion in FP equation** remains isotropic: `∂m/∂t - (σ²/2)Δm + ... = 0`

**Examples**: `examples/advanced/anisotropic_crowd_dynamics_2d/`

## Future Enhancement: Full Matrix Diffusion

For **fully anisotropic diffusion** where the Fokker-Planck equation itself has direction-dependent diffusion:

$$
\frac{\partial m}{\partial t} - \nabla \cdot (D(x) \nabla m) + \nabla \cdot (m v) = 0
$$

where $D(x) \in \mathbb{R}^{d \times d}$ is a symmetric positive-definite diffusion matrix.

### Mathematical Formulation

**SDE**:
$$
dX_t = v(X_t, t)dt + \sqrt{D(X_t)} dW_t
$$

**FP Equation**:
$$
\frac{\partial m}{\partial t} = \nabla \cdot (D \nabla m) - \nabla \cdot (m v)
$$

**HJB Equation** (with matrix diffusion):
$$
-\frac{\partial u}{\partial t} + H(x, m, \nabla u, t) - \frac{1}{2}\text{tr}(D \nabla^2 u) = 0
$$

### API Design Proposal

```python
# Option 1: Extend diffusion_coeff to support matrix output
def matrix_diffusion(x):
    """Return diffusion matrix D(x) at position x"""
    # Example: Corridor-like domain
    if in_corridor(x):
        return np.array([[1.0, 0.0],    # Easy along corridor
                        [0.0, 0.1]])    # Hard perpendicular
    else:
        return 0.1 * np.eye(2)          # Isotropic elsewhere

problem = MFGProblem(
    ...,
    diffusion_coeff=matrix_diffusion,  # Returns matrix, not scalar
)

# Option 2: Separate parameter for matrix diffusion
problem = MFGProblem(
    ...,
    diffusion_coeff=0.1,               # Fallback scalar
    diffusion_matrix_func=matrix_diffusion,  # Optional matrix D(x)
)
```

### Implementation Challenges

1. **Type System Complexity**:
   - Need to distinguish scalar vs matrix returns from Callable
   - Type hint: `Callable[[NDArray], float | NDArray]` is ambiguous

2. **Solver Compatibility**:
   - FDM solvers need to handle `div(D∇m)` instead of `σ²Δm`
   - GFDM needs matrix-weighted least squares
   - Particle methods need Cholesky decomposition of D for sampling

3. **Validation**:
   - Must verify D(x) is symmetric positive-definite at all points
   - Numerical stability depends on condition number of D

4. **Performance**:
   - Matrix operations are more expensive than scalar
   - Need efficient sparse matrix representations

### Use Cases

This enhancement would enable:

1. **Anisotropic Brownian Motion**: Particles diffuse faster in certain directions
2. **Geological Flows**: Permeability varies by direction (porous media)
3. **Molecular Dynamics**: Non-uniform thermal noise
4. **Traffic Networks**: Lane-dependent diffusion

### Implementation Strategy

**Phase 1**: Extend type system and protocol
```python
# In base_problem.py
sigma: float | Callable  # Current: Callable returns float
                         # Future: Callable returns float | NDArray

# Add helper method
def get_diffusion_at(self, x) -> float | NDArray:
    """Get diffusion at position x (scalar or matrix)"""
    if callable(self.sigma):
        result = self.sigma(x)
        if isinstance(result, np.ndarray):
            # Validate matrix
            assert result.shape == (self.dimension, self.dimension)
            assert is_symmetric_positive_definite(result)
        return result
    return self.sigma
```

**Phase 2**: Update FP solvers
- Modify `fp_fdm.py` to handle div(D∇m)
- Update `fp_particle.py` to use Cholesky(D) for sampling

**Phase 3**: Update HJB solvers
- Modify diffusion term from `σ²Δu` to `tr(D∇²u)`
- Semi-Lagrangian needs D for backward characteristics

**Phase 4**: Documentation and examples
- Theory doc: Matrix diffusion mathematical formulation
- Example: Corridor evacuation with anisotropic diffusion

### Decision: Not Implementing Now

**Rationale**:
- Current scalar + Hamiltonian approach covers existing use cases
- Matrix diffusion adds significant complexity
- No immediate user demand
- Can be added incrementally without breaking changes

**Recommendation**: Track as future enhancement, implement if user requests arise.

## References

[^1]: Anisotropic MFG Theory: `docs/theory/applications/anisotropic_mfg_mathematical_formulation.md`
[^2]: Current Implementation: `mfg_pde/core/base_problem.py:201` (diffusion_coeff parameter)
[^3]: Existing Anisotropic Examples: `examples/advanced/anisotropic_crowd_dynamics_2d/`

---

**Conclusion**: Current implementation provides **position-dependent scalar diffusion** via Callable. Full **matrix diffusion** is documented here as a future enhancement if needed.
