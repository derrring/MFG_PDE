# Smoothing Kernels: Mathematical Formulation

**Module**: `mfg_pde.utils.numerical.smoothing_kernels`
**Date**: 2025-11-04
**Status**: ✅ COMPLETE

---

## Table of Contents

1. [Mathematical Definitions](#mathematical-definitions)
2. [Kernel Properties](#kernel-properties)
3. [Differential Operators on Kernels](#differential-operators-on-kernels)
4. [Mollifiers and Regularization](#mollifiers-and-regularization)
5. [Convolution and Approximation](#convolution-and-approximation)
6. [Kernel Types](#kernel-types)
7. [Applications in MFG_PDE](#applications-in-mfg_pde)
8. [References](#references)

---

## 1. Mathematical Definitions

### 1.1 Kernel Function

A **kernel function** $K: \mathbb{R}^d \to \mathbb{R}$ is a non-negative, integrable function used for smoothing, interpolation, and approximation.

**Scaled Kernel**:
$$K_h(\mathbf{x}) = \frac{1}{h^d} K\left(\frac{\mathbf{x}}{h}\right)$$

where:
- $\mathbf{x} \in \mathbb{R}^d$ is the evaluation point
- $h > 0$ is the **smoothing length** (bandwidth, support radius)
- $d$ is the spatial dimension

**Radial Kernels** (isotropic):
For radial kernels, $K(\mathbf{x}) = k(\|\mathbf{x}\|)$ depends only on distance $r = \|\mathbf{x}\|$:
$$K_h(r) = \frac{1}{h^d} k\left(\frac{r}{h}\right)$$

where $k: [0, \infty) \to \mathbb{R}$ is the **kernel profile function**.

**Normalization Form**:
Define normalized variable $q = r/h$, then:
$$K_h(r) = \frac{1}{h^d} k(q), \quad q = \frac{r}{h}$$

This form is used in the `smoothing_kernels` module:
```python
kernel(r, h) = (1/h^d) * k(r/h)
```

### 1.2 Normalization Condition

A properly normalized kernel satisfies:
$$\int_{\mathbb{R}^d} K_h(\mathbf{x}) \, d\mathbf{x} = 1, \quad \forall h > 0$$

For radial kernels in dimension $d$:
$$\int_{\mathbb{R}^d} K_h(r) \, d\mathbf{x} = \omega_d \int_0^\infty K_h(r) r^{d-1} \, dr = 1$$

where $\omega_d = \frac{2\pi^{d/2}}{\Gamma(d/2)}$ is the surface area of the unit sphere in $\mathbb{R}^d$:
- $\omega_1 = 2$ (1D)
- $\omega_2 = 2\pi$ (2D)
- $\omega_3 = 4\pi$ (3D)

**Example**: Gaussian kernel
$$K_h(r) = \frac{1}{(2\pi h^2)^{d/2}} \exp\left(-\frac{r^2}{2h^2}\right)$$
satisfies $\int_{\mathbb{R}^d} K_h(\mathbf{x}) \, d\mathbf{x} = 1$.

---

## 2. Kernel Properties

A **good** kernel for numerical methods should satisfy:

### 2.1 Essential Properties

1. **Non-negativity**:
   $$K_h(\mathbf{x}) \geq 0, \quad \forall \mathbf{x} \in \mathbb{R}^d$$

2. **Normalization** (partition of unity):
   $$\int_{\mathbb{R}^d} K_h(\mathbf{x}) \, d\mathbf{x} = 1$$

3. **Symmetry** (for radial kernels):
   $$K_h(\mathbf{x}) = K_h(-\mathbf{x})$$

4. **Compact support** or **rapid decay**:
   - **Compact**: $K_h(\mathbf{x}) = 0$ for $\|\mathbf{x}\| > R h$ (support radius $R$)
   - **Infinite support**: $K_h(\mathbf{x}) \to 0$ exponentially fast as $\|\mathbf{x}\| \to \infty$

5. **Smoothness**:
   $K_h \in C^n(\mathbb{R}^d)$ for desired regularity $n$

### 2.2 Desirable Properties

6. **Monotonicity** (for radial kernels):
   $$\frac{d}{dr} K_h(r) \leq 0, \quad r \geq 0$$

7. **Positive definiteness**:
   For Wendland kernels, guarantees stability in RBF interpolation.

8. **Moment properties**:
   $$\int_{\mathbb{R}^d} \mathbf{x}^\alpha K_h(\mathbf{x}) \, d\mathbf{x} = 0, \quad |\alpha| = 1, 2, \ldots, p$$
   for $p$th-order accuracy.

### 2.3 Dirac Delta Approximation

As $h \to 0$, $K_h(\mathbf{x})$ approximates the Dirac delta distribution:
$$\lim_{h \to 0} K_h(\mathbf{x}) = \delta(\mathbf{x})$$

in the distributional sense:
$$\lim_{h \to 0} \int_{\mathbb{R}^d} f(\mathbf{x}) K_h(\mathbf{x} - \mathbf{y}) \, d\mathbf{x} = f(\mathbf{y})$$

for all continuous functions $f$ with compact support.

---

## 3. Differential Operators on Kernels

For radial kernels $K(\mathbf{x}) = k(r)$ where $r = \|\mathbf{x}\|$, the differential operators have explicit forms. These formulas are essential for GFDM, SPH, and other meshfree methods that compute derivatives via kernel approximations.

### 3.1 Gradient of Radial Kernel

For a radial kernel $K(\mathbf{x}) = k(r)$ with $r = \|\mathbf{x}\|$:

$$\nabla K(\mathbf{x}) = \frac{dk}{dr} \frac{\mathbf{x}}{r} = k'(r) \frac{\mathbf{x}}{r}$$

**Scaled kernel** with smoothing length $h$:
$$K_h(\mathbf{x}) = \frac{1}{h^d} k\left(\frac{r}{h}\right)$$

$$\nabla K_h(\mathbf{x}) = \frac{1}{h^{d+1}} k'(q) \frac{\mathbf{x}}{r}, \quad q = \frac{r}{h}$$

**Component form** in $\mathbb{R}^d$:
$$\frac{\partial K_h}{\partial x_i} = \frac{1}{h^{d+1}} k'(q) \frac{x_i}{r}$$

### 3.2 Divergence of Radial Kernel

The divergence of a vector field $\mathbf{F}(\mathbf{x}) = F(r) \mathbf{x}/r$ is:

$$\nabla \cdot \left( F(r) \frac{\mathbf{x}}{r} \right) = F'(r) + \frac{d-1}{r} F(r)$$

For the gradient of a radial kernel:
$$\nabla \cdot (\nabla K) = \Delta K = k''(r) + \frac{d-1}{r} k'(r)$$

**Scaled kernel**:
$$\Delta K_h(\mathbf{x}) = \frac{1}{h^{d+2}} \left[ k''(q) + \frac{d-1}{q h} k'(q) \right], \quad q = \frac{r}{h}$$

Alternatively, in terms of the radial variable $q$:
$$\Delta K_h(\mathbf{x}) = \frac{1}{h^{d+2}} \left[ k''(q) + \frac{d-1}{q} k'(q) \right]$$

### 3.3 Laplacian of Radial Kernel

The Laplacian $\Delta K = \nabla^2 K = \nabla \cdot (\nabla K)$ for radial kernel $K(r) = k(r)$:

$$\Delta K(r) = k''(r) + \frac{d-1}{r} k'(r)$$

**In $d$ dimensions**:
- **1D**: $\Delta K = k''(r)$
- **2D**: $\Delta K = k''(r) + \frac{1}{r} k'(r)$
- **3D**: $\Delta K = k''(r) + \frac{2}{r} k'(r)$

**Scaled kernel** with smoothing length $h$:
$$\Delta K_h(r) = \frac{1}{h^{d+2}} \left[ k''(q) + \frac{d-1}{q} k'(q) \right], \quad q = \frac{r}{h}$$

### 3.4 Applications in Numerical Methods

**SPH Approximations**:
- Gradient approximation: $\nabla f(\mathbf{x}) \approx \sum_j f_j \nabla_{\mathbf{x}} K_h(\mathbf{x} - \mathbf{x}_j)$
- Laplacian approximation: $\Delta f(\mathbf{x}) \approx \sum_j f_j \Delta_{\mathbf{x}} K_h(\mathbf{x} - \mathbf{x}_j)$

**GFDM Weight Functions**:
- The gradient $\nabla_{\mathbf{x}} K_h(\mathbf{x} - \mathbf{x}_i)$ provides directional weights for computing partial derivatives
- Used in weighted least squares reconstruction: `mfg_pde.utils.numerical.gfdm_operators`

**Kernel Derivatives in Code**:
The `smoothing_kernels` module provides `evaluate_with_derivative()` method returning both $k(q)$ and $k'(q)$, which can be used to construct $\nabla K$ and $\Delta K$ using the formulas above.

---

## 4. Mollifiers and Regularization

### 4.1 Mollifier Definition

A **mollifier** is a smooth, compactly supported, non-negative kernel satisfying:
$$\phi \in C^\infty_c(\mathbb{R}^d), \quad \phi \geq 0, \quad \int_{\mathbb{R}^d} \phi(\mathbf{x}) \, d\mathbf{x} = 1$$

Standard mollifier (bump function):
$$\phi(\mathbf{x}) = \begin{cases}
C \exp\left(\frac{-1}{1 - \|\mathbf{x}\|^2}\right), & \|\mathbf{x}\| < 1 \\
0, & \|\mathbf{x}\| \geq 1
\end{cases}$$

where $C$ is a normalization constant.

**Scaled Mollifier**:
$$\phi_\epsilon(\mathbf{x}) = \frac{1}{\epsilon^d} \phi\left(\frac{\mathbf{x}}{\epsilon}\right)$$

### 4.2 Mollification (Regularization)

Given a function $f \in L^1_{\text{loc}}(\mathbb{R}^d)$, its **mollification** is:
$$f_\epsilon(\mathbf{x}) = (f * \phi_\epsilon)(\mathbf{x}) = \int_{\mathbb{R}^d} f(\mathbf{y}) \phi_\epsilon(\mathbf{x} - \mathbf{y}) \, d\mathbf{y}$$

**Properties**:
1. $f_\epsilon \in C^\infty(\mathbb{R}^d)$ (infinitely smooth)
2. $f_\epsilon \to f$ in $L^p$ as $\epsilon \to 0$ (for $f \in L^p$)
3. $\nabla (f_\epsilon) = f * \nabla \phi_\epsilon$ (differentiation under convolution)

**Use in MFG**: Regularize non-smooth initial data $m_0$ to ensure $m_\epsilon \in C^\infty$.

### 4.3 Relationship to Smoothing Kernels

- **Mollifiers**: Always $C^\infty$ and compactly supported
- **Smoothing Kernels**: May have infinite support (Gaussian) or finite smoothness (Wendland C²)

Example mapping:
- **Wendland C⁶** ≈ Mollifier (C⁶ smooth, compact support)
- **Gaussian** ≈ Not a mollifier (infinite support, but $C^\infty$ smooth)

---

## 5. Convolution and Approximation

### 5.1 Continuous Convolution

For kernel $K_h$ and function $f$:
$$(f * K_h)(\mathbf{x}) = \int_{\mathbb{R}^d} f(\mathbf{y}) K_h(\mathbf{x} - \mathbf{y}) \, d\mathbf{y}$$

**Interpretation**: Weighted average of $f$ in neighborhood of $\mathbf{x}$ with weights $K_h$.

**Key Property**:
If $\int K_h = 1$, then $(f * K_h)(\mathbf{x}) \to f(\mathbf{x})$ as $h \to 0$ (pointwise for continuous $f$).

### 5.2 Kernel Density Estimation (KDE)

Given particles $\{\mathbf{x}_i\}_{i=1}^N$, approximate density:
$$\rho_h(\mathbf{x}) = \frac{1}{N} \sum_{i=1}^N K_h(\mathbf{x} - \mathbf{x}_i)$$

**Used in MFG_PDE**: `FPParticleSolver` in hybrid mode uses KDE to project particle density onto grid.

### 5.3 SPH Approximation

In Smoothed Particle Hydrodynamics (SPH), approximate function $f$ at $\mathbf{x}$:
$$f(\mathbf{x}) \approx \sum_{j=1}^N \frac{m_j}{\rho_j} f(\mathbf{x}_j) K_h(\mathbf{x} - \mathbf{x}_j)$$

where:
- $m_j$ = particle mass
- $\rho_j$ = particle density
- $K_h$ = SPH kernel (typically cubic or quintic spline)

**Gradient Approximation**:
$$\nabla f(\mathbf{x}) \approx \sum_{j=1}^N \frac{m_j}{\rho_j} f(\mathbf{x}_j) \nabla K_h(\mathbf{x} - \mathbf{x}_j)$$

### 5.4 GFDM Weighted Least Squares

In Generalized Finite Difference Method (GFDM), derivatives approximated via weighted least squares with kernel weights:
$$w_{ij} = K_h(\mathbf{x}_i - \mathbf{x}_j)$$

Minimize weighted residual:
$$\min_{\nabla f} \sum_j w_{ij} \left[ f(\mathbf{x}_j) - f(\mathbf{x}_i) - \nabla f \cdot (\mathbf{x}_j - \mathbf{x}_i) \right]^2$$

**Used in MFG_PDE**: `gfdm_operators.py` uses Gaussian RBF weights for GFDM derivative approximation.

---

## 6. Kernel Types

### 6.1 Gaussian Kernel (Infinite Support)

**Profile Function**:
$$k(q) = \exp(-q^2), \quad q = \frac{r}{h}$$

**Scaled Kernel** (dimension-independent normalization):
$$K_h(r) = \frac{1}{(\pi h^2)^{d/2}} \exp\left(-\frac{r^2}{h^2}\right)$$

**Properties**:
- $C^\infty$ smooth
- Infinite support (practically zero for $r > 3h$)
- Fourier transform: Gaussian (ideal low-pass filter)

**MFG_PDE Implementation**: `GaussianKernel()` in `smoothing_kernels.py`

### 6.2 Wendland Kernels (Compact Support)

**General Parameterized Form**:
The Wendland family of kernels has the structure:
$$k(q) = \begin{cases}
(1-q)^m P_k(q), & 0 \leq q < 1 \\
0, & q \geq 1
\end{cases}$$

where:
- $m = 2k + 2$ determines the support power
- $P_k(q)$ is a polynomial of degree $k$ ensuring $C^{2k}$ continuity
- $k \in \{0, 1, 2, 3\}$ is the **smoothness parameter**

**Polynomial Coefficients by Smoothness Order**:

**Wendland C⁰** ($k=0$, $m=2$):
$$k(q) = (1-q)^2, \quad 0 \leq q < 1$$
- C⁰ continuous, $P_0(q) = 1$

**Wendland C²** ($k=1$, $m=4$):
$$k(q) = (1-q)^4 (4q + 1), \quad 0 \leq q < 1$$
- C² continuous, $P_1(q) = 4q + 1$

**Wendland C⁴** ($k=2$, $m=6$):
$$k(q) = (1-q)^6 (35q^2 + 18q + 3), \quad 0 \leq q < 1$$
- C⁴ continuous, $P_2(q) = 35q^2 + 18q + 3$

**Wendland C⁶** ($k=3$, $m=8$):
$$k(q) = (1-q)^8 (32q^3 + 25q^2 + 8q + 1), \quad 0 \leq q < 1$$
- C⁶ continuous, $P_3(q) = 32q^3 + 25q^2 + 8q + 1$

**Derivative Formula**:
$$\frac{dk}{dq} = \begin{cases}
(1-q)^{m-1} \left[-m P_k(q) + (1-q) P_k'(q)\right], & 0 \leq q < 1 \\
0, & q \geq 1
\end{cases}$$

**Properties**:
- Compact support: $[0, h]$
- Positive definite (guaranteed for all Wendland kernels)
- Minimal degree polynomial for given smoothness
- Higher $k$ → smoother but computationally more expensive

**Normalization** (dimension-dependent):
Wendland kernels require normalization constants $\sigma_d$:
$$K_h(r) = \frac{\sigma_d}{h^d} k\left(\frac{r}{h}\right)$$

**MFG_PDE Implementation**: `WendlandKernel(k, dimension)` with $k \in \{0, 1, 2, 3\}$
- Unified parameterized class replaces separate C0/C2/C4/C6 classes
- Automatic polynomial coefficient generation based on $k$
- Factory: `create_kernel('wendland_c2')` → `WendlandKernel(k=1)`

### 6.3 Cubic Spline (B-Spline M4)

**General Theory**:
B-splines are piecewise polynomials constructed via recursive convolution. The cubic spline (order 4, degree 3) is the most widely used SPH kernel.

**Profile Function**:
$$k(q) = \begin{cases}
1 - \frac{3}{2}q^2 + \frac{3}{4}q^3, & 0 \leq q < 1 \\
\frac{1}{4}(2 - q)^3, & 1 \leq q < 2 \\
0, & q \geq 2
\end{cases}$$

**Derivative**:
$$\frac{dk}{dq} = \begin{cases}
-3q + \frac{9}{4}q^2, & 0 \leq q < 1 \\
-\frac{3}{4}(2 - q)^2, & 1 \leq q < 2 \\
0, & q \geq 2
\end{cases}$$

**Normalization Constants** (dimension-dependent):
- 1D: $\sigma = \frac{2}{3}$
- 2D: $\sigma = \frac{10}{7\pi} \approx 0.4547$
- 3D: $\sigma = \frac{1}{\pi} \approx 0.3183$

**Scaled Kernel**:
$$K_h(r) = \frac{\sigma}{h^d} k\left(\frac{r}{h}\right)$$

**Properties**:
- C² continuous (second derivative continuous)
- Compact support: $[0, 2h]$
- Positive everywhere in support
- Monotonically decreasing
- Bell-shaped profile
- SPH standard (Monaghan & Lattanzio, 1985)

**Mathematical Justification**:
The cubic spline is the B-spline of order 4, obtained by convolving the box function with itself 4 times. It provides a good balance between:
- Smoothness (C² continuity)
- Compact support (radius 2h)
- Computational efficiency
- Numerical stability

**MFG_PDE Implementation**: `CubicSplineKernel(dimension=d)`

### 6.4 Quintic Spline (B-Spline M6)

**General Theory**:
The quintic spline (order 6, degree 5) provides higher accuracy and smoother derivatives than the cubic spline, at the cost of larger support radius.

**Profile Function**:
$$k(q) = \begin{cases}
(3-q)^5 - 6(2-q)^5 + 15(1-q)^5, & 0 \leq q < 1 \\
(3-q)^5 - 6(2-q)^5, & 1 \leq q < 2 \\
(3-q)^5, & 2 \leq q < 3 \\
0, & q \geq 3
\end{cases}$$

**Expanded Forms**:
- Region 1 ($0 \leq q < 1$): $66 - 60q^2 + 30q^3 - 10q^4 + q^5$
- Region 2 ($1 \leq q < 2$): $51 - 75q + 60q^2 - 30q^3 + 9q^4 - q^5$
- Region 3 ($2 \leq q < 3$): $243 - 405q + 270q^2 - 90q^3 + 15q^4 - q^5$

**Derivative**:
$$\frac{dk}{dq} = \begin{cases}
-120q + 90q^2 - 40q^3 + 5q^4, & 0 \leq q < 1 \\
-75 + 120q - 90q^2 + 36q^3 - 5q^4, & 1 \leq q < 2 \\
-405 + 540q - 270q^2 + 60q^3 - 5q^4, & 2 \leq q < 3 \\
0, & q \geq 3
\end{cases}$$

**Normalization Constants** (dimension-dependent):
- 1D: $\sigma = \frac{1}{120}$
- 2D: $\sigma = \frac{7}{478\pi} \approx 0.00466$
- 3D: $\sigma = \frac{1}{120\pi} \approx 0.00265$

**Scaled Kernel**:
$$K_h(r) = \frac{\sigma}{h^d} k\left(\frac{r}{h}\right)$$

**Properties**:
- C⁴ continuous (fourth derivative continuous)
- Compact support: $[0, 3h]$
- Positive everywhere in support
- Monotonically decreasing
- Smoother than cubic spline
- Higher computational cost (larger support)
- Better accuracy for gradient approximations

**Mathematical Justification**:
The quintic spline is the B-spline of order 6, providing:
- Higher smoothness (C⁴ vs C² for cubic)
- More accurate particle approximations
- Better derivative estimates
- Reduced numerical noise in SPH simulations

However, the larger support radius (3h vs 2h) increases computational cost by ~3× in 2D and ~4× in 3D.

**Trade-off Analysis**:
| Property | Cubic Spline | Quintic Spline |
|:---------|:-------------|:---------------|
| Continuity | C² | C⁴ |
| Support | 2h | 3h |
| Neighbors (2D) | ~12-16 | ~28-36 |
| Accuracy | Good | Excellent |
| Cost | Low | Medium |

**MFG_PDE Implementation**: `QuinticSplineKernel(dimension=d)`

**Note on Implementation**: B-splines are kept as separate classes (cubic and quintic) rather than parameterized by degree because:
1. Only these two orders are standard in SPH literature
2. Normalization constants are well-established
3. Higher-order B-splines require complex recursive formulas
4. Parameterization would add significant complexity for minimal gain

### 5.5 Simple Polynomial Kernels

**Cubic Kernel**:
$$k(q) = \begin{cases}
(1-q)^3, & 0 \leq q < 1 \\
0, & q \geq 1
\end{cases}$$

**Properties**:
- C⁰ continuous (discontinuous derivative at q=1)
- Compact support: $[0, h]$
- Simplest polynomial kernel
- Fast evaluation
- No dimension-dependent normalization needed for relative weights

**Quartic Kernel**:
$$k(q) = \begin{cases}
(1-q)^4, & 0 \leq q < 1 \\
0, & q \geq 1
\end{cases}$$

**Properties**:
- C¹ continuous (continuous first derivative)
- Compact support: $[0, h]$
- Slightly smoother than cubic
- Fast evaluation

**Use Cases**:
These simple kernels are useful for:
- GFDM weight functions where only relative magnitudes matter
- Quick prototyping
- Educational examples
- Cases where C^n continuity is not critical

**MFG_PDE Implementation**: `CubicKernel()`, `QuarticKernel()`

---

## 7. Applications in MFG_PDE

### 6.1 Kernel Density Estimation (KDE)

**Module**: `mfg_pde.alg.numerical.fp_solvers.fp_particle`

**Usage**: Project particle distribution onto grid
```python
# Hybrid mode: particles → KDE → grid
M_grid[t, x] = Σ_i w_i * K_h(x - x_i)
```

**Kernel**: Typically Gaussian RBF

### 6.2 GFDM Spatial Derivatives

**Module**: `mfg_pde.utils.numerical.gfdm_operators`

**Usage**: Weight neighbors in least squares for derivative approximation
```python
# Weights for GFDM
w_ij = gaussian_rbf_weight(r_ij, h)
```

**Kernel**: Gaussian RBF (currently), can use Wendland for compact support

### 6.3 Semi-Lagrangian Interpolation

**Module**: `mfg_pde.alg.numerical.hjb_solvers.hjb_semi_lagrangian`

**Usage**: Interpolate along characteristics
```python
# Shepard's method with RBF weights
u_interp = Σ_i w_i u_i / Σ_i w_i
```

**Kernel**: Gaussian or Wendland RBF

### 6.4 SPH (Future)

**Potential Module**: `mfg_pde.alg.numerical.sph_solvers` (not yet implemented)

**Usage**: Smoothed Particle Hydrodynamics for fluid dynamics
```python
# Density estimation
ρ_i = Σ_j m_j K_h(x_i - x_j)
```

**Kernel**: Cubic or quintic spline (SPH standard)

---

## 8. References

### 7.1 Kernel Theory

1. **Wendland, H.** "Piecewise polynomial, positive definite and compactly supported radial functions of minimal degree." *Advances in Computational Mathematics* 4.1 (1995): 389-396.
   - Original Wendland kernels paper

2. **Adams, R. A., Fournier, J. J. F.** *Sobolev Spaces*. Academic Press (2003).
   - Mollifiers and regularization theory

3. **Evans, L. C.** *Partial Differential Equations*. 2nd edition, American Mathematical Society (2010).
   - Section 2.3: Mollifiers and weak derivatives

### 7.2 SPH and Particle Methods

4. **Monaghan, J. J.** "Smoothed particle hydrodynamics." *Reports on Progress in Physics* 68.8 (2005): 1703.
   - Comprehensive SPH review, kernel requirements

5. **Liu, G. R., Liu, M. B.** *Smoothed Particle Hydrodynamics: A Meshfree Particle Method.* World Scientific (2003).
   - Textbook on SPH kernels

6. **Dehnen, W., Aly, H.** "Improving convergence in smoothed particle hydrodynamics simulations without pairing instability." *Monthly Notices of the Royal Astronomical Society* 425.2 (2012): 1068-1082.
   - Modern Wendland kernel analysis for SPH

### 7.3 GFDM

7. **Benito, J. J., et al.** "Influence of several factors in the generalized finite difference method." *Applied Mathematical Modelling* 25.12 (2001): 1039-1053.
   - GFDM with RBF weights

8. **Gavete, L., et al.** "Improvements of generalized finite difference method and comparison with other meshless method." *Applied Mathematical Modelling* 27.10 (2003): 831-847.
   - GFDM kernel weight analysis

### 7.4 Kernel Density Estimation

9. **Silverman, B. W.** *Density Estimation for Statistics and Data Analysis*. Chapman and Hall (1986).
   - Classic KDE reference, bandwidth selection

10. **Scott, D. W.** *Multivariate Density Estimation: Theory, Practice, and Visualization*. Wiley (2015).
    - Modern KDE theory

---

## Summary

This document provides the mathematical foundation for the `smoothing_kernels` module in MFG_PDE. Key concepts:

1. **Kernels** are normalized, non-negative functions for smoothing and approximation
2. **Mollifiers** are $C^\infty$ compactly supported kernels for regularization
3. **Convolution** with kernels provides approximations converging to identity
4. **Kernel types** trade off smoothness vs. compact support:
   - Gaussian: $C^\infty$, infinite support
   - Wendland: $C^{2k}$, compact support
   - Splines: $C^{2}$ (cubic) or $C^{4}$ (quintic), compact support
5. **Applications** span KDE, GFDM, SPH, and semi-Lagrangian methods

All kernels in the module satisfy normalization $\int K_h = 1$ and converge to Dirac delta as $h \to 0$, making them suitable for numerical PDE methods.

**Code Reference**: `mfg_pde/utils/numerical/smoothing_kernels.py:1-850`
