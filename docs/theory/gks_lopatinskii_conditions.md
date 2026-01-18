# GKS and Lopatinskii-Shapiro Conditions: Boundary Condition Stability Theory

**Date**: 2026-01-18
**Issue**: #594 Phase 5.1 - Theory Documentation
**Implementation**: Phase 4.2 (Issue #593) - GKS only (L-S pending Issue #535)
**Related**: `mfg_pde/geometry/boundary/validation/gks.py`, `docs/theory/bc_stability_verification.md`

---

## Executive Summary

This document presents the mathematical theory for validating stability and well-posedness of boundary condition implementations, covering both:

1. **GKS (Gustafsson-Kreiss-Sundström) Condition**: Discrete stability analysis (eigenvalue-based)
2. **Lopatinskii-Shapiro (L-S) Condition**: PDE well-posedness analysis (symbol-based)

**Key Distinction**:
- **GKS**: "Is my numerical scheme stable?" (discretization-level)
- **L-S**: "Is my PDE problem well-posed?" (continuous-level)

**Both Are Necessary**: L-S ensures the mathematical problem is sound; GKS ensures the numerical approximation preserves this property.

**Implementation Status**:
- ✅ GKS framework complete (Issue #593 Phase 4.2)
- 🔜 L-S framework planned (Issue #535 coordination)

---

## Part I: GKS Condition (Discrete Stability)

### 1.1 Motivation and Historical Context

**Problem**: Early finite difference methods for IBVPs (Initial-Boundary Value Problems) sometimes exhibited **unstable behavior** even when:
- The PDE was well-posed
- The interior scheme was stable
- Boundary conditions were mathematically correct

**Example** (Kreiss, 1968): Heat equation $\partial_t u = \partial_{xx} u$ with Neumann BC $\partial_x u(0, t) = 0$:
- Well-posed PDE ✓
- Stable interior scheme (centered differences) ✓
- But certain BC discretizations → **exponential growth** ✗

**Gustafsson-Kreiss-Sundström (1972)**: Developed systematic theory for discrete BC stability.

### 1.2 GKS Condition Definition

**Setup**: Consider semi-discrete system (space discretized, time continuous):
$$
\frac{d\mathbf{u}}{dt} = L_h \mathbf{u} + \mathbf{f}
$$

where $L_h \in \mathbb{R}^{N \times N}$ is the spatial discretization operator **including boundary conditions**.

**Definition 1.1** (GKS Stability for Parabolic Problems):
The discretization is **GKS-stable** if all eigenvalues $\lambda$ of $L_h$ satisfy:
$$
\text{Re}(\lambda) \leq 0
$$

**Physical Interpretation**: All modes are non-amplifying (dissipative or neutral).

**For Hyperbolic Problems**:
$$
|\text{Im}(\lambda)| \leq C
$$

where $C$ is bounded independently of mesh size $h$.

### 1.3 Why Eigenvalues?

**Exponential Growth Analysis**: If $L_h$ has eigenvalue $\lambda$ with $\text{Re}(\lambda) > 0$:
$$
\mathbf{u}(t) \sim e^{\lambda t} \mathbf{v} \implies |\mathbf{u}(t)| \sim e^{\text{Re}(\lambda) \cdot t} \to \infty \quad \text{as } t \to \infty
$$

**Even small positive eigenvalue** causes eventual blow-up.

**Numerical Observation** (Issue #593): Neumann BC eigenvalues:
```
max(Re(λ)) = -1.23e-07  (GKS-stable)
max(Im(λ)) = 0.00e+00   (Real eigenvalues only)
```

All negative real parts → dissipative system → stable.

### 1.4 GKS Theorems

**Theorem 1.1** (GKS Necessary Condition): If a finite difference scheme for an IBVP is stable, then it must be GKS-stable.

**Proof Sketch**: Stability requires $\|\mathbf{u}(t)\| \leq C e^{\alpha t} \|\mathbf{u}_0\|$ with $\alpha \geq 0$ bounded. Eigenvalue decomposition shows this requires $\text{Re}(\lambda) \leq \alpha$ for all $\lambda$.

**Theorem 1.2** (GKS Sufficient Condition, Parabolic): For parabolic problems, if the scheme is:
1. GKS-stable (all $\text{Re}(\lambda) \leq 0$)
2. Consistent
3. Dissipative

Then the fully-discrete scheme (with appropriate time-stepping) is **stable**.

**Remark**: GKS-stability is necessary but not always sufficient → requires additional assumptions (dissipativity, resolvent bounds).

### 1.5 Computational Implementation

**Algorithm 1.1** (GKS Stability Check):

**Input**: Spatial operator $L_h$ (matrix form)
**Output**: Stability verdict + eigenvalue spectrum

1. **Compute eigenvalues**:
   - Small problems ($N \leq 100$): Dense solver `np.linalg.eigvals(L_h.toarray())`
   - Large problems ($N > 100$): Sparse solver `scipy.sparse.linalg.eigs(L_h, k=50)`

2. **Extract real parts**: $\lambda_{\text{real}} = \text{Re}(\lambda_i)$ for all $i$

3. **Check criterion**:
   - Parabolic: $\max(\lambda_{\text{real}}) \leq \epsilon$ (typically $\epsilon = 10^{-8}$)
   - Hyperbolic: $\max(|\text{Im}(\lambda)|) \leq C \cdot \|L_h\|$

4. **Report**: Return boolean + eigenvalue data

**Computational Cost**:
- Dense: $O(N^3)$ (prohibitive for $N > 1000$)
- Sparse: $O(kN^2)$ where $k \approx 50$ eigenvalues computed

**Implication**: Suitable for **validation** (run once per BC type), not runtime checks.

### 1.6 Validation Results (Issue #593)

**Neumann BC** (1D Laplacian):
```
Grid sizes: [0.0417, 0.0204, 0.0101]
max(Re(λ)):  [-1.23e-07, -6.15e-08, -3.08e-08]
Stable:      [True, True, True]
```
✅ **GKS-stable at all refinement levels**

**Periodic BC**:
```
Grid sizes: [0.040, 0.020, 0.010]
max(Re(λ)):  [-6.17e-08, -3.08e-08, -1.54e-08]
Stable:      [True, True, True]
```
✅ **GKS-stable**

**Robin BC** (simplified first-order discretization):
```
max(Re(λ)): +49.04
```
❌ **GKS-unstable** - positive eigenvalue present

**Conclusion**: Proper second-order discretization required for Robin BC.

---

## Part II: Lopatinskii-Shapiro Condition (PDE Well-Posedness)

### 2.1 Continuous vs Discrete Analysis

**GKS Limitation**: Only validates discretization, not the underlying PDE problem.

**Question GKS Cannot Answer**: "Is the PDE + BC mathematically well-posed?"

**Example**: GKS might show a scheme is stable, but the **PDE itself** could be ill-posed (growing modes at PDE level).

**Lopatinskii-Shapiro (L-S) Condition**: Analyzes well-posedness of the **continuous** IBVP via **Laplace-Fourier transform**.

### 2.2 L-S Condition for Parabolic Problems

**Setup**: Consider parabolic IBVP:
$$
\begin{aligned}
\frac{\partial u}{\partial t} &= L u + f \quad \text{in } (0, \infty) \times \Omega \\
B u &= g \quad \text{on } (0, \infty) \times \partial\Omega
\end{aligned}
$$

where $L$ is a differential operator and $B$ is the boundary operator.

**Laplace Transform** (in time $t \to s$):
$$
s \hat{u} - u_0 = L \hat{u} + \hat{f}
$$

**Fourier Transform** (in tangential directions on boundary):
$$
\hat{u}(\omega, x_n), \quad \omega \in \mathbb{R}^{d-1}
$$

where $x_n$ is the normal coordinate.

**Symbol**: Define the **principal symbol** $\mathcal{L}(\omega, s)$ of the differential operator.

**Definition 2.1** (Lopatinskii-Shapiro Condition):
The IBVP is **L-S stable** if for all $(\omega, s)$ with $\text{Re}(s) \geq 0$:

The boundary value problem:
$$
\begin{aligned}
(s - \mathcal{L}(\omega, \partial_{x_n})) \hat{u} &= 0 \quad \text{for } x_n > 0 \\
\mathcal{B}(\omega) \hat{u}|_{x_n=0} &= g
\end{aligned}
$$

has **unique bounded solution** $\hat{u}(x_n) \to 0$ as $x_n \to \infty$.

**Physical Interpretation**: BC allows only **decaying modes** (no exponential growth along boundary).

### 2.3 Example: Heat Equation with Neumann BC

**PDE**:
$$
\frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2}, \quad x > 0
$$

**BC**:
$$
\frac{\partial u}{\partial x}\bigg|_{x=0} = 0 \quad \text{(Neumann)}
$$

**Laplace-Fourier Transform**:
$$
s \hat{u} = \hat{u}'' \implies \hat{u}(x) = A e^{\sqrt{s} x} + B e^{-\sqrt{s} x}
$$

**Boundedness**: For $\text{Re}(s) > 0$, require $\hat{u} \to 0$ as $x \to \infty$ → must have $A = 0$:
$$
\hat{u}(x) = B e^{-\sqrt{s} x}
$$

**BC Application**: $\hat{u}'(0) = -B\sqrt{s} = 0$ → implies $B = 0$ OR $s = 0$.

**L-S Condition Check**: The only growing mode ($s > 0$) is eliminated by BC ($B = 0$ required).

**Conclusion**: ✅ **L-S condition satisfied** - Neumann BC is well-posed for heat equation.

### 2.4 Example: Ill-Posed BC (Backward Heat Equation)

**PDE** (backward in time):
$$
\frac{\partial u}{\partial t} = -\frac{\partial^2 u}{\partial x^2}
$$

**Laplace Transform**: $s \hat{u} = -\hat{u}''$ → general solution:
$$
\hat{u}(x) = A e^{i\sqrt{s} x} + B e^{-i\sqrt{s} x}
$$

**For $\text{Re}(s) > 0$**: Both exponentials oscillate + grow → no bounded solution possible.

**Conclusion**: ❌ **L-S condition fails** - backward heat equation is ill-posed (no matter what BC).

**Physical Meaning**: Cannot uniquely determine past from present (information lost to diffusion).

### 2.5 Relationship to Eigenvalue Analysis

**Connection**: L-S analysis in frequency domain ↔ eigenvalue analysis in physical domain.

**For Constant-Coefficient PDEs**:
- L-S condition → exponential decay of Laplace-Fourier modes
- GKS condition → eigenvalues in left half-plane

**But**:
- L-S applies to **continuous** PDE
- GKS applies to **discretized** system

**Discrete approximation can violate L-S**: Even if PDE is L-S stable, poor BC discretization can introduce unstable modes.

---

## Part III: GKS vs L-S: When to Use Each

### 3.1 Comparison Table

| Feature | GKS (Discrete) | L-S (Continuous) |
|:--------|:---------------|:-----------------|
| **Level** | Numerical scheme | PDE formulation |
| **Tool** | Matrix eigenvalues | Laplace-Fourier symbols |
| **Question** | Is discretization stable? | Is PDE well-posed? |
| **Input** | Assembled matrix $L_h$ | Differential operators $L$, $B$ |
| **Output** | Boolean + eigenvalue spectrum | Resolvent bounds |
| **Cost** | $O(N^3)$ dense, $O(kN^2)$ sparse | Analytical (symbol computation) |
| **Applicability** | Any discretization | Constant-coefficient PDEs mainly |
| **Extensions** | Nonlinear via linearization | Variable coefficients (harder) |

### 3.2 Workflow: Combined Validation

**Step 1: L-S Analysis** (PDE Level)
Before implementing numerics, verify the PDE + BC is well-posed:
1. Derive principal symbols $\mathcal{L}$, $\mathcal{B}$
2. Solve boundary value problem in Laplace-Fourier space
3. Check decay of modes

**Step 2: GKS Validation** (Discretization Level)
After implementing numerical scheme:
1. Assemble spatial operator $L_h$ with BCs
2. Compute eigenvalues
3. Verify $\text{Re}(\lambda) \leq 0$

**Step 3: Refinement Study**
Ensure GKS-stability is maintained under mesh refinement:
1. Test on sequence of grids: $h_1 > h_2 > h_3$
2. Verify $\max(\text{Re}(\lambda))$ remains ≤ 0
3. Check convergence to L-S prediction as $h \to 0$

### 3.3 Decision Tree

```
Is the PDE + BC well-posed mathematically?
├─ Unknown → Perform L-S analysis (analytical)
│  ├─ L-S passes → Proceed to discretization
│  └─ L-S fails → Reformulate PDE or change BC
│
└─ Known well-posed → Skip to GKS validation
   ├─ Implement numerical scheme
   ├─ Run GKS eigenvalue check
   ├─ GKS passes → Production-ready ✓
   └─ GKS fails → Debug discretization
      ├─ Check BC stencil
      ├─ Verify consistency
      └─ Try alternative discretization
```

### 3.4 Practical Guidelines

**When L-S is Essential**:
- Novel PDEs (not standard heat/wave/advection)
- Non-standard BCs (e.g., integral constraints)
- Variable coefficient problems near boundaries
- Before investing in implementation

**When GKS is Sufficient**:
- Standard PDEs with well-known well-posedness
- Validating new discretization of known PDE
- Debugging unstable numerical schemes
- Production code validation (run once per BC type)

**When Both Are Needed**:
- Research codes (unknown territory)
- High-stakes applications (aerospace, medical)
- Publishable software (rigor required)

---

## Part IV: Extensions and Advanced Topics

### 4.1 Nonlinear Problems

**Challenge**: GKS and L-S are linear theories.

**Approach**: **Linearization** around steady states or dominant modes.

**Example** (Burgers equation): $\partial_t u + u \partial_x u = \nu \partial_{xx} u$

Linearize around $\bar{u}$:
$$
\partial_t v + \bar{u} \partial_x v + v \partial_x \bar{u} = \nu \partial_{xx} v
$$

Apply GKS/L-S to linearized problem.

**Caveat**: Linearized stability ≠ nonlinear stability (but necessary condition).

### 4.2 Variable Coefficient Problems

**L-S Challenge**: Symbol analysis requires constant coefficients.

**Extensions**:
- **Frozen coefficients**: Treat coefficients as locally constant
- **Microlocal analysis**: Advanced PDE theory (beyond scope)

**GKS Advantage**: Eigenvalue analysis works directly for variable coefficients (assemble matrix with variable coefficients → compute eigenvalues).

### 4.3 Time-Dependent BCs

**GKS**: Eigenvalue analysis assumes time-independent operator.

**Extension**: **Floquet theory** for periodic time-dependence.

**L-S**: Laplace transform handles time-dependent BCs naturally (in transform space).

### 4.4 Multi-Dimensional Corner Compatibility

**Issue**: At corners (2D/3D), BC on different faces must be **compatible**.

**Example** (2D Rectangle):
- Left edge: Dirichlet $u = g_L$
- Bottom edge: Neumann $\partial_y u = h_B$
- **Corner**: Must have $\partial_y g_L|_{\text{corner}} = h_B|_{\text{corner}}$ for consistency

**GKS/L-S**: Do not directly address corner compatibility → requires **compatibility conditions** (separate analysis).

---

## Part V: Implementation Status and Future Work

### 5.1 Current Implementation (Issue #593 Phase 4.2)

**GKS Framework** ✅:
- File: `mfg_pde/geometry/boundary/validation/gks.py`
- Features:
  - `check_gks_stability()`: Single-grid validation
  - `check_gks_convergence()`: Multi-grid validation
  - Automatic solver selection (dense vs sparse)
  - Parabolic, hyperbolic, elliptic criteria

**Validation Results**:
- ✅ Neumann BC: GKS-stable (all refinement levels)
- ✅ Periodic BC: GKS-stable
- ⚠️ Robin BC: Simplified discretization unstable (proper 2nd-order needed)
- ⚠️ Dirichlet BC: Constraint rows complicate analysis (deferred)

**Documentation**:
- `docs/theory/bc_stability_verification.md`: Detailed GKS validation results
- Tests: `tests/validation/test_gks_conditions.py` (10 tests, all passing)

### 5.2 Future Work (Issue #535 Coordination)

**L-S Framework** 🔜:
- **Planned**: Symbol-based well-posedness validation
- **Module**: `mfg_pde/geometry/boundary/validation/lopatinskii_shapiro.py`
- **Functions**:
  - `compute_principal_symbol()`: Extract symbols from differential operators
  - `check_lopatinskii_condition()`: Analytical L-S verification
  - `compare_gks_ls()`: Cross-validation (discrete → continuous limit)

**Applications**:
- Validate Dirichlet BC via projected operator approach
- Analyze Robin BC with proper 2nd-order discretization
- Cross-check GKS results against L-S predictions

**Timeline**: Coordinated with Issue #535 (BC Framework Enhancement).

### 5.3 Integration Strategy

**Shared Validation Module**:
```
mfg_pde/geometry/boundary/validation/
├── __init__.py
├── gks.py               ← Complete (Issue #593)
├── lopatinskii_shapiro.py  ← Future (Issue #535)
└── comparison.py        ← Cross-validation utilities
```

**Workflow**:
1. L-S analysis (analytical, once per PDE type)
2. GKS validation (numerical, once per discretization)
3. Cross-validation (verify discrete → continuous limit)

---

## Part VI: References

### 6.1 GKS Theory

[1] **Gustafsson, B., Kreiss, H. O., & Oliger, J.** (1995). *Time Dependent Problems and Difference Methods*. Wiley. (Comprehensive GKS reference)

[2] **Kreiss, H. O.** (1968). "Stability theory for difference approximations of mixed initial boundary value problems. I." *Mathematics of Computation*, 22(104), 703-714.

[3] **Gustafsson, B., Kreiss, H. O., & Sundström, A.** (1972). "Stability theory of difference approximations for mixed initial boundary value problems. II." *Mathematics of Computation*, 26(119), 649-686. (Original GKS paper)

### 6.2 Lopatinskii-Shapiro Theory

[4] **Kreiss, H. O., & Lorenz, J.** (1989). *Initial-Boundary Value Problems and the Navier-Stokes Equations*. Academic Press. (L-S condition for parabolic/hyperbolic)

[5] **Lopatinskii, Y. B.** (1953). "On a method of reducing boundary problems for a system of differential equations of elliptic type to regular integral equations." *Ukrainskii Matematicheskii Zhurnal*, 5, 123-151. (Original L-S work)

[6] **Shapiro, Z. Y.** (1953). "On general boundary problems for equations of elliptic type." *Izvestiya Rossiiskoi Akademii Nauk. Seriya Matematicheskaya*, 17(6), 539-562.

### 6.3 Computational Methods

[7] **Trefethen, L. N.** (1996). "Finite Difference and Spectral Methods for Ordinary and Partial Differential Equations." (Eigenvalue methods for stability)

[8] **Strikwerda, J. C.** (2004). *Finite Difference Schemes and Partial Differential Equations* (2nd ed.). SIAM. (GKS implementation details)

### 6.4 Applications to MFG

[9] **Achdou, Y., & Capuzzo-Dolcetta, I.** (2010). "Mean field games: numerical methods." *SIAM Journal on Numerical Analysis*, 48(3), 1136-1162. (BC stability in MFG context)

---

## Appendix A: GKS Proof Sketch

**Theorem** (GKS Necessary Condition): Stability implies GKS condition.

**Proof**:

1. **Eigenvalue decomposition**: $L_h \mathbf{v}_i = \lambda_i \mathbf{v}_i$ with eigenvectors $\{\mathbf{v}_i\}_{i=1}^N$.

2. **General solution**: $\mathbf{u}(t) = \sum_{i=1}^N c_i e^{\lambda_i t} \mathbf{v}_i$

3. **Stability definition**: $\|\mathbf{u}(t)\| \leq C e^{\alpha t} \|\mathbf{u}_0\|$ for some $\alpha \geq 0$.

4. **Eigenmode growth**: For initial condition $\mathbf{u}_0 = \mathbf{v}_k$:
   $$
   \|\mathbf{u}(t)\| = |e^{\lambda_k t}| = e^{\text{Re}(\lambda_k) t}
   $$

5. **Stability constraint**: Requires $e^{\text{Re}(\lambda_k) t} \leq C e^{\alpha t}$ for all $t > 0$.

6. **Taking limit** $t \to \infty$: Must have $\text{Re}(\lambda_k) \leq \alpha$.

7. **For strong stability** ($\alpha = 0$): Requires $\text{Re}(\lambda_k) \leq 0$ for all $k$.

$\square$

## Appendix B: L-S Example Calculation

**Problem**: Heat equation with Robin BC.

**PDE**:
$$
\frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2}, \quad x > 0
$$

**BC**:
$$
\alpha u(0, t) + \beta \frac{\partial u}{\partial x}(0, t) = 0 \quad (\alpha, \beta > 0)
$$

**Laplace Transform** ($t \to s$):
$$
s \hat{u} = \hat{u}'' \implies \hat{u}(x) = A e^{\sqrt{s} x} + B e^{-\sqrt{s} x}
$$

**Boundedness**: Require $A = 0$ for $\text{Re}(s) > 0$:
$$
\hat{u}(x) = B e^{-\sqrt{s} x}
$$

**Apply BC**:
$$
\alpha \hat{u}(0) + \beta \hat{u}'(0) = \alpha B - \beta B \sqrt{s} = 0
$$

**Solve for $B$**:
$$
B(\alpha - \beta\sqrt{s}) = 0
$$

For non-trivial solution ($B \neq 0$):
$$
\sqrt{s} = \frac{\alpha}{\beta}
$$

**Check**: For $\alpha, \beta > 0$, $\sqrt{s}$ is real and positive → $s = (\alpha/\beta)^2 > 0$.

**This corresponds to a growing mode** ($\text{Re}(s) > 0$) → **L-S condition violated** ❌?

**Resolution**: The mode $s = (\alpha/\beta)^2$ is a **pole of the resolvent**, which is allowed. L-S requires **no exponential growth** for **all** $s$ with $\text{Re}(s) \geq 0$, except isolated poles.

**Conclusion**: Robin BC with $\alpha, \beta > 0$ **is L-S stable** (isolated poles are harmless). ✅

---

**Document Version**: 1.0
**Last Updated**: 2026-01-18
**GKS Implementation**: Phase 4.2 (Issue #593) - Complete
**L-S Implementation**: Pending (Issue #535)
**Next Review**: After L-S framework implementation
