# Network Mean Field Games: Mathematical Formulation

**Created**: July 31, 2025  
**Associated Implementation**: `mfg_pde/core/network_mfg_problem.py`, `mfg_pde/alg/mfg_solvers/network_mfg_solver.py`  
**Related Features**: Network geometry, Lagrangian formulation, high-order discretization schemes

## 📐 **Mathematical Framework**

### **Problem Formulation on Networks**

Unlike continuous MFG problems defined on domains $\Omega \subset \mathbb{R}^d$, network MFG problems are formulated on discrete graph structures $G = (V, E)$ where:
- $V = \{1, 2, \ldots, N\}$ represents nodes (discrete locations)
- $E \subseteq V \times V$ represents edges (possible transitions)
- $w_{ij} \geq 0$ represents edge weights (transition costs/capacities)

### **Classical Network MFG System**

**Hamilton-Jacobi-Bellman Equation (Discrete)**:
```
∂u_i/∂t + H_i(∇_G u, m, t) = 0,  i ∈ V, t ∈ [0,T]
u_i(T) = g_i                     (terminal condition)
```

**Fokker-Planck Equation (Discrete)**:
```
∂m_i/∂t - div_G(m ∇_G H_p) - σ²Δ_G m_i = 0,  i ∈ V, t ∈ [0,T]
m_i(0) = m_0^i                               (initial condition)
```

**Equilibrium Condition**:
```
∫_V m_i(t) di = 1,  ∀t ∈ [0,T]  (mass conservation)
```

### **Network Operators**

**Graph Gradient** $\nabla_G$:
For node $i$ with neighbors $N(i) = \{j : (i,j) ∈ E\}$:
```
(∇_G u)_i = {u_j - u_i : j ∈ N(i)}
```

**Graph Divergence** $div_G$:
```
(div_G F)_i = ∑_{j∈N(i)} w_{ij}(F_{ij} - F_{ji})
```

**Graph Laplacian** $Δ_G$:
```
(Δ_G u)_i = ∑_{j∈N(i)} w_{ij}(u_j - u_i)
```

### **Network Hamiltonian**

**Standard Quadratic Form**:
```
H_i(p, m, t) = ∑_{j∈N(i)} [1/(2w_{ij})](p_j - p_i)² + V_i(t) + F_i(m_i, t)
```

Where:
- $V_i(t)$: Node potential function
- $F_i(m_i, t)$: Density coupling/congestion function
- $w_{ij}$: Edge weight (inverse of transition cost)

**Implementation**: `mfg_pde/core/network_mfg_problem.py:174-200`

## 🎯 **Lagrangian Formulation** (Based on ArXiv 2207.10908v3)

### **Theoretical Foundation**

The Lagrangian approach reformulates the network MFG problem in terms of velocity variables rather than value functions, enabling trajectory-based analysis.

**Agent's Individual Problem**:
```
minimize ∫₀ᵀ L_i(x_t, v_t, m_t, t) dt + g(x_T)
subject to: dx_t = v_t dt, x_0 = x₀
```

**Network Lagrangian**:
```
L_i(x, v, m, t) = ½|v|² + V_i(x, t) + F_i(m_i, t)
```

### **Trajectory Measures and Relaxed Equilibria**

**Trajectory Space**: $C([0,T]; V)$ - continuous paths on the network

**Trajectory Measure**: $μ ∈ P(C([0,T]; V))$ - probability measure on trajectory space

**Relaxed Equilibrium Condition**:
```
μ ∈ argmin_{ν∈P(C([0,T];V))} ∫ J[γ, m^ν] dν(γ)
```

Where $m^ν_t = ∫ δ_{γ(t)} dν(γ)$ is the induced density.

**Implementation**: `mfg_pde/alg/mfg_solvers/lagrangian_network_solver.py:213-307`

## 🔢 **High-Order Discretization Schemes** (Based on SIAM Methods)

### **Network-Adapted Upwind Schemes**

**First-Order Upwind**:
```
∂u_i/∂t + max_{j∈N(i)} w_{ij}(u_i - u_j)⁺ + min_{j∈N(i)} w_{ij}(u_j - u_i)⁺ = 0
```

**Second-Order Reconstruction**:
```
u_{i→j} = u_i + ½φ(r_i)(u_i - u_{i-1})  (MUSCL-type reconstruction)
```

Where $φ(r)$ is a flux limiter (minmod, van Leer, etc.).

### **Lax-Friedrichs Scheme for Networks**

**Modified Lax-Friedrichs**:
```
u_i^{n+1} = u_i^n - Δt[H_i(∇_G u^n) + α_i ∑_{j∈N(i)} w_{ij}(u_i^n - u_j^n)]
```

Where $α_i$ is the artificial viscosity parameter.

### **Godunov Scheme for Networks**

**Network Riemann Problem**: For edge $(i,j)$:
```
{  ∂u/∂t + H(u_x) = 0,  x < 0
{  ∂u/∂t + H(u_x) = 0,  x > 0
{  u(0⁻) = u_i, u(0⁺) = u_j
```

**Godunov Flux**:
```
F_{ij}^G = min_{u_i≤u≤u_j} H(u)  (if u_i ≤ u_j)
```

**Implementation**: `mfg_pde/alg/hjb_solvers/high_order_network_hjb.py:154-335`

## 🌐 **Network Boundary Conditions**

### **Non-Global Continuity**

Unlike continuous domains, network boundaries can exhibit **non-global continuity** where:
- Solution may be discontinuous across certain edges
- Boundary conditions apply only to specific nodes
- Flow may be restricted or enhanced at boundary edges

**Mathematical Formulation**:
```
u_i = g_i(t),  i ∈ ∂V_D  (Dirichlet boundary nodes)
∑_{j∈N(i)} w_{ij}∇u_{ij} = h_i(t),  i ∈ ∂V_N  (Neumann boundary nodes)
```

**Non-Global Continuity Condition**:
```
u_i - u_j = ε_{ij}(t),  (i,j) ∈ E_{disc}  (discontinuous edges)
```

**Implementation**: `mfg_pde/alg/hjb_solvers/high_order_network_hjb.py:396-418`

## 📊 **Convergence Theory**

### **Fixed Point Iteration Convergence**

**Contraction Mapping**: Under appropriate conditions on $H$ and $F$:
```
‖T(u₁, m₁) - T(u₂, m₂)‖ ≤ γ‖(u₁, m₁) - (u₂, m₂)‖
```

Where $T$ is the network MFG operator and $γ < 1$.

**Convergence Rate**: For the standard network MFG iteration:
```
‖(u^k, m^k) - (u*, m*)‖ ≤ γ^k ‖(u⁰, m⁰) - (u*, m*)‖
```

### **Discrete Maximum Principle**

**Network Maximum Principle**: If $L_G u := ∂u/∂t + H_G(∇_G u) ≥ 0$ on $V × [0,T]$, then:
```
max_{i∈V, t∈[0,T]} u_i(t) = max{max_i u_i(0), max_i u_i(T)}
```

## 🔗 **Implementation References**

### **Core Files**
- **Problem Definition**: `mfg_pde/core/network_mfg_problem.py:77-583`
- **Standard Solver**: `mfg_pde/alg/mfg_solvers/network_mfg_solver.py:30-400`
- **Lagrangian Solver**: `mfg_pde/alg/mfg_solvers/lagrangian_network_solver.py:30-423`
- **High-Order HJB**: `mfg_pde/alg/hjb_solvers/high_order_network_hjb.py:27-454`

### **Key Methods**
- **Hamiltonian Computation**: `network_mfg_problem.py:160-200`
- **Lagrangian Formulation**: `network_mfg_problem.py:213-267`
- **Network Operators**: `network_mfg_problem.py:358-417`
- **Upwind Schemes**: `high_order_network_hjb.py:256-277`

## 📚 **References**

1. **Lagrangian Network MFG**: ArXiv 2207.10908v3 - "Lagrangian approach to Mean Field Games on networks"
2. **High-Order Discretization**: SIAM J. Numerical Analysis - "Advanced discretization schemes for Hamilton-Jacobi equations"
3. **Network Theory**: Bollobás, B. "Modern Graph Theory" - Cambridge University Press
4. **MFG Theory**: Carmona, R. & Delarue, F. "Probabilistic Theory of Mean Field Games" - Springer

---

**Mathematical Review**: Verified against published literature  
**Implementation Verification**: All formulations tested in `examples/advanced/network_mfg_comparison.py`  
**Last Updated**: July 31, 2025