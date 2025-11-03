# Adaptive Parameter Selection for MFG Problems

**Date**: 2025-11-03
**Purpose**: Design adaptive/intelligent selection of key MFG parameters
**Status**: Design proposal

---

## Motivation

MFG problems have several critical parameters that significantly affect convergence, stability, and accuracy:

1. **σ (sigma)** - Diffusion coefficient / noise intensity
2. **λ (coefCT)** - Coupling strength
3. **θ (damping_factor)** - Picard iteration damping
4. **Δt, Δx** - Discretization parameters
5. **Tolerance** - Convergence thresholds

Currently, these are **manually specified** by users, requiring expertise and trial-and-error.

**Goal**: Develop **adaptive selection methods** that automatically choose appropriate values based on problem characteristics.

---

## Current State

### **What Exists**

1. **Adaptive Damping Factor** (Implemented):
   - Location: `docs/development/analysis/hybrid_damping_factor_analysis.md`
   - Considers: T, coefCT, sigma
   - Method: Heuristic formula based on problem parameters

2. **Fixed Parameters** (User-specified):
   - σ, coefCT: Set at problem creation
   - No automatic adjustment

3. **Adaptive Mesh Refinement** (Design exists):
   - Spatial discretization adaptation
   - Not parameter adaptation

### **What's Missing**

- ❌ Adaptive sigma selection
- ❌ Adaptive coupling strength (coefCT)
- ❌ Joint parameter optimization
- ❌ Problem-specific parameter tuning
- ❌ Runtime parameter adjustment

---

## Literature Review: Adaptive Parameters in PDEs

### **1. Adaptive Diffusion Coefficient**

#### **Anisotropic Diffusion** (Perona-Malik 1990)
```
σ(∇u) = g(|∇u|)  where g(s) = 1 / (1 + (s/K)²)
```
- Reduces diffusion at edges (large gradients)
- Preserves sharp features
- **Application to MFG**: Reduce σ where density gradients steep

#### **Adaptive Diffusivity** (Weickert 1998)
```
σ(x,t) = σ₀ · h(density_variance(x,t))
```
- Lower σ in regions with rapid density changes
- Higher σ in smooth regions
- **Application to MFG**: Spatially-varying σ based on m(x,t)

#### **Entropy-Based Adaptation** (Guermond et al. 2011)
```
σ = σ_max · (S - S_min) / (S_max - S_min)
where S = entropy residual
```
- High diffusion where entropy production high
- Low diffusion where solution smooth
- **Application to MFG**: σ based on HJB residual

### **2. CFL-Based Time Step Selection**

#### **Standard CFL Condition**
```
Δt ≤ CFL · min(Δx² / σ, Δx / |v_max|)
```
- Stability-based time step
- **Application to MFG**: Automatic Δt from σ, domain size

#### **Adaptive Time Stepping** (Gustafsson 1994)
```
Δt_{n+1} = Δt_n · (tol / error_n)^(1/(p+1))
where p = order of method
```
- Based on local truncation error
- **Application to MFG**: Adjust Δt based on iteration error

### **3. Parameter Identification Methods**

#### **Inverse Problem Approach**
- Estimate σ from observed data
- Minimize ||m_computed - m_observed||
- **Application to MFG**: Calibrate σ from empirical density

#### **Cross-Validation**
- Split data → train → validate
- Choose σ minimizing validation error
- **Application to MFG**: σ selection for data-driven MFG

---

## Proposed Methods

### **Method 1: Adaptive Sigma Based on Problem Characteristics**

#### **Heuristic Formula**

```python
def compute_adaptive_sigma(
    domain_size: float,
    time_horizon: float,
    coupling_strength: float,
    initial_density_variance: float,
    problem_type: str = "standard"
) -> float:
    """
    Compute appropriate sigma based on problem characteristics.

    Principles:
    - Larger domain → larger sigma (more diffusion needed)
    - Longer time → smaller sigma (less diffusion per step)
    - High coupling → smaller sigma (coupling dominates)
    - High density variance → larger sigma (smooth out variations)

    Parameters
    ----------
    domain_size : float
        Spatial extent: max(x_max - x_min for each dimension)
    time_horizon : float
        Total time T
    coupling_strength : float
        Coupling coefficient λ (coefCT)
    initial_density_variance : float
        Var(m_0(x))
    problem_type : str
        "standard", "congestion", "evacuation", etc.

    Returns
    -------
    sigma : float
        Recommended diffusion coefficient
    """
    # Base sigma from dimensional analysis
    # [sigma] = L²/T for diffusion equation
    base_sigma = domain_size**2 / time_horizon

    # Scale by problem characteristics
    coupling_factor = max(0.1, 1.0 - 10 * coupling_strength)  # Less σ if high coupling
    variance_factor = np.sqrt(initial_density_variance)  # More σ if high variance

    # Problem-type adjustments
    type_factors = {
        "standard": 1.0,
        "congestion": 0.5,  # Less diffusion, coupling dominates
        "evacuation": 1.5,  # More diffusion, agents spread out
        "lq": 0.3,  # Low diffusion for LQ problems
    }
    type_factor = type_factors.get(problem_type, 1.0)

    sigma = base_sigma * coupling_factor * variance_factor * type_factor

    # Clamp to reasonable range
    return np.clip(sigma, 0.01, 10.0)
```

#### **Usage**

```python
from mfg_pde.utils.adaptive_parameters import compute_adaptive_sigma

# User specifies problem, not sigma
problem = MFGProblem(
    xmin=0, xmax=1, Nx=100,
    T=1.0, Nt=50,
    coefCT=0.5,
    components=components
)

# Automatically compute sigma
sigma = compute_adaptive_sigma(
    domain_size=1.0,
    time_horizon=1.0,
    coupling_strength=0.5,
    initial_density_variance=compute_variance(problem.m0),
    problem_type="congestion"
)

problem.sigma = sigma  # Apply computed value
```

---

### **Method 2: Spatially-Varying Sigma (Anisotropic Diffusion)**

#### **Concept**

Instead of constant σ, use σ(x,t) that adapts to local density features.

#### **Implementation**

```python
class AdaptiveSigmaField:
    """Spatially and temporally varying diffusion coefficient."""

    def __init__(
        self,
        sigma_min: float = 0.1,
        sigma_max: float = 2.0,
        gradient_threshold: float = 0.1
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.K = gradient_threshold

    def compute_sigma(self, m: NDArray, dx: float) -> NDArray:
        """
        Compute σ(x,t) based on current density m(x,t).

        Uses edge-preserving diffusion:
        σ(x) = σ_min + (σ_max - σ_min) / (1 + (|∇m| / K)²)

        Parameters
        ----------
        m : NDArray
            Current density field
        dx : float
            Spatial discretization

        Returns
        -------
        sigma_field : NDArray
            Spatially-varying sigma
        """
        # Compute density gradient magnitude
        grad_m = np.gradient(m, dx)
        grad_magnitude = np.sqrt(sum(g**2 for g in grad_m))

        # Edge-preserving diffusion (Perona-Malik style)
        sigma_field = self.sigma_min + (self.sigma_max - self.sigma_min) / (
            1 + (grad_magnitude / self.K)**2
        )

        return sigma_field

    def update(self, m: NDArray, dx: float, problem: MFGProblem):
        """Update problem's sigma field."""
        problem.sigma_field = self.compute_sigma(m, dx)
```

#### **Integration with Solver**

```python
class HJBFDMSolver:
    def solve(self, ...):
        # ... existing code ...

        if hasattr(self.problem, 'adaptive_sigma'):
            # Update sigma based on current density
            sigma_field = self.problem.adaptive_sigma.compute_sigma(m_current, dx)
            # Use spatially-varying sigma in diffusion term
        else:
            # Use constant sigma
            sigma_field = self.problem.sigma
```

---

### **Method 3: Learning-Based Sigma Selection**

#### **Concept**

Train a model to predict optimal σ from problem features.

#### **Training Data**

Collect pairs: (problem features, optimal sigma)
- Features: domain_size, T, coefCT, m0_variance, etc.
- Optimal sigma: Determined by convergence studies

#### **Model**

```python
class SigmaPredictor:
    """ML model for sigma prediction."""

    def __init__(self):
        # Simple neural network or random forest
        self.model = None  # Load pre-trained model

    def predict(
        self,
        domain_size: float,
        time_horizon: float,
        coupling_strength: float,
        density_features: dict
    ) -> float:
        """
        Predict sigma from problem characteristics.

        Returns
        -------
        sigma : float
            Predicted diffusion coefficient
        confidence : float
            Prediction confidence (0-1)
        """
        features = np.array([
            domain_size,
            time_horizon,
            coupling_strength,
            density_features['variance'],
            density_features['skewness'],
            density_features['num_modes']
        ])

        sigma = self.model.predict(features)
        confidence = self.model.predict_confidence(features)

        return sigma, confidence
```

---

### **Method 4: Runtime Adaptive Sigma**

#### **Concept**

Adjust σ during iteration based on convergence behavior.

#### **Algorithm**

```python
class AdaptivePicardIterator:
    """Picard iterator with adaptive parameter adjustment."""

    def __init__(
        self,
        problem: MFGProblem,
        hjb_solver: HJBSolver,
        fp_solver: FPSolver,
        adaptive_sigma: bool = True,
        adaptive_damping: bool = True
    ):
        self.problem = problem
        self.hjb_solver = hjb_solver
        self.fp_solver = fp_solver
        self.adaptive_sigma = adaptive_sigma
        self.adaptive_damping = adaptive_damping

        # Initial values
        self.sigma_history = []
        self.error_history = []

    def solve(self, max_iterations: int = 100, tol: float = 1e-6):
        """Solve with adaptive parameter adjustment."""

        for iteration in range(max_iterations):
            # Solve HJB and FP
            u_new = self.hjb_solver.solve()
            m_new = self.fp_solver.solve()

            # Compute error
            error = self.compute_error(u_new, m_new)
            self.error_history.append(error)

            # Adapt sigma if enabled
            if self.adaptive_sigma and iteration > 5:
                self.problem.sigma = self._adjust_sigma(
                    current_sigma=self.problem.sigma,
                    error_history=self.error_history,
                    iteration=iteration
                )
                self.sigma_history.append(self.problem.sigma)

            # Adapt damping if enabled
            if self.adaptive_damping:
                damping = self._adjust_damping(error_history=self.error_history)

            # Apply damping
            u_new = (1 - damping) * u_new + damping * u_old
            m_new = (1 - damping) * m_new + damping * m_old

            if error < tol:
                break

        return u_new, m_new

    def _adjust_sigma(
        self,
        current_sigma: float,
        error_history: list[float],
        iteration: int
    ) -> float:
        """
        Adjust sigma based on convergence behavior.

        Principles:
        - If error oscillating → increase sigma (more smoothing)
        - If error decreasing steadily → keep sigma
        - If error stagnant → try reducing sigma (less over-smoothing)
        """
        if len(error_history) < 3:
            return current_sigma

        recent_errors = error_history[-3:]

        # Check for oscillation (sign changes in error difference)
        diffs = np.diff(recent_errors)
        oscillating = (diffs[0] * diffs[1] < 0)  # Sign change

        # Check for stagnation (very slow decrease)
        stagnant = (recent_errors[-1] / recent_errors[-3] > 0.95)

        if oscillating:
            # Increase sigma to smooth out oscillations
            new_sigma = current_sigma * 1.2
        elif stagnant:
            # Decrease sigma to reduce over-smoothing
            new_sigma = current_sigma * 0.9
        else:
            # Converging well, keep sigma
            new_sigma = current_sigma

        # Clamp to reasonable range
        return np.clip(new_sigma, 0.01, 10.0)
```

---

### **Method 5: Coupled Parameter Optimization**

#### **Concept**

Jointly optimize σ, coefCT, damping_factor using Bayesian optimization.

#### **Objective**

Minimize: convergence_time + accuracy_penalty

```python
def objective(params: dict) -> float:
    """Objective for parameter optimization."""
    sigma, coefCT, damping = params['sigma'], params['coefCT'], params['damping']

    # Solve problem with these parameters
    problem = MFGProblem(sigma=sigma, coefCT=coefCT, ...)
    solver = PicardIterator(problem, damping_factor=damping)
    result = solver.solve()

    # Compute objective
    convergence_time = result.num_iterations
    accuracy = result.final_error

    # Weighted combination
    return convergence_time + 100 * accuracy  # Weight accuracy heavily
```

#### **Optimization**

```python
from scipy.optimize import minimize

# Bayesian optimization or grid search
best_params = minimize(
    objective,
    x0=[1.0, 0.5, 0.5],  # Initial guess [sigma, coefCT, damping]
    bounds=[(0.01, 10), (0, 2), (0, 1)],
    method='L-BFGS-B'
)
```

---

## Implementation Plan

### **Phase 1: Heuristic Adaptive Sigma** (Immediate)

1. Implement `compute_adaptive_sigma()` function
2. Add to `mfg_pde/utils/adaptive_parameters.py`
3. Document usage patterns
4. Add examples showing automatic vs manual sigma

**Effort**: 2-3 days
**Impact**: High (immediate usability improvement)

### **Phase 2: Spatially-Varying Sigma** (Short-term)

1. Implement `AdaptiveSigmaField` class
2. Modify solvers to support σ(x,t)
3. Add anisotropic diffusion examples
4. Validate against known problems

**Effort**: 1-2 weeks
**Impact**: Medium (advanced feature for specific problems)

### **Phase 3: Runtime Adaptive Parameters** (Medium-term)

1. Implement `AdaptivePicardIterator`
2. Add sigma adjustment logic
3. Test on convergence-sensitive problems
4. Compare with fixed-parameter baseline

**Effort**: 2-3 weeks
**Impact**: Medium-High (automatic tuning during solve)

### **Phase 4: Learning-Based Selection** (Long-term/Research)

1. Collect training data (problem → optimal params)
2. Train prediction model
3. Integrate into problem creation workflow
4. Validate on held-out problems

**Effort**: 1-2 months
**Impact**: High (fully automatic parameter selection)

---

## Example Usage

### **Example 1: Automatic Sigma**

```python
from mfg_pde import MFGProblem
from mfg_pde.utils.adaptive_parameters import compute_adaptive_sigma

# Define problem WITHOUT specifying sigma
problem = MFGProblem(
    xmin=0, xmax=1, Nx=100,
    T=1.0, Nt=50,
    coefCT=0.5
)

# Automatically compute sigma
sigma = compute_adaptive_sigma(
    domain_size=problem.domain_size,
    time_horizon=problem.T,
    coupling_strength=problem.coefCT,
    initial_density_variance=problem.m0_variance,
    problem_type="congestion"
)

print(f"Recommended sigma: {sigma:.3f}")
problem.sigma = sigma
```

### **Example 2: Spatially-Varying Sigma**

```python
from mfg_pde.utils.adaptive_parameters import AdaptiveSigmaField

# Create adaptive sigma field
adaptive_sigma = AdaptiveSigmaField(
    sigma_min=0.1,
    sigma_max=2.0,
    gradient_threshold=0.1
)

problem.adaptive_sigma = adaptive_sigma

# Solver automatically uses σ(x,t)
solver = HJBFDMSolver(problem)
result = solver.solve()  # Uses spatially-varying sigma
```

### **Example 3: Runtime Adaptation**

```python
from mfg_pde.solvers import AdaptivePicardIterator

solver = AdaptivePicardIterator(
    problem,
    hjb_solver=HJBFDMSolver(problem),
    fp_solver=FPParticleSolver(problem),
    adaptive_sigma=True,  # Enable sigma adaptation
    adaptive_damping=True  # Enable damping adaptation
)

result = solver.solve()

# Inspect adaptation history
import matplotlib.pyplot as plt
plt.plot(solver.sigma_history)
plt.xlabel('Iteration')
plt.ylabel('Sigma')
plt.title('Sigma Evolution During Solving')
plt.show()
```

---

## Benefits

### **For Users**

- ✅ **Less expertise required**: Don't need to know "good" sigma values
- ✅ **Faster setup**: Automatic parameter selection
- ✅ **Better convergence**: Problem-appropriate parameters
- ✅ **Fewer failed runs**: Adaptive adjustment prevents divergence

### **For Developers**

- ✅ **Reduced support burden**: Fewer "why doesn't my problem converge?" questions
- ✅ **Better benchmarks**: Consistent parameter selection
- ✅ **Research opportunities**: Learning-based approaches

---

## Related Work

### **In MFG Literature**

1. **Achdou & Capuzzo-Dolcetta (2010)**: Fixed σ, manual tuning
2. **Carlini & Silva (2014)**: σ from data (inverse problem)
3. **Ruthotto et al. (2020)**: Neural network-based σ learning

### **In PDE Methods**

1. **Perona-Malik (1990)**: Anisotropic diffusion
2. **Weickert (1998)**: Tensor-valued diffusion
3. **Gustafsson (1994)**: Adaptive time stepping

### **In Machine Learning**

1. **Bayesian Optimization**: Hyperparameter tuning
2. **AutoML**: Automatic model selection
3. **Neural Architecture Search**: Structure optimization

---

## Risks and Mitigation

### **Risk 1: Over-Automation**

**Risk**: Users lose understanding of parameters

**Mitigation**:
- Document what adaptive methods do
- Always allow manual override
- Log selected parameters
- Provide "explain" mode showing reasoning

### **Risk 2: Suboptimal Selection**

**Risk**: Heuristics choose poor parameters

**Mitigation**:
- Extensive validation on benchmark problems
- Confidence intervals on predictions
- Fallback to conservative defaults
- User feedback mechanism

### **Risk 3: Computational Overhead**

**Risk**: Adaptation adds significant cost

**Mitigation**:
- Lightweight heuristics (negligible cost)
- Cache computations
- Only adapt when beneficial (convergence issues)
- Amortize cost over iterations

---

## Summary

### **Proposed Adaptive Methods**

1. **Heuristic sigma** - Problem-characteristic-based formula
2. **Spatially-varying sigma** - σ(x,t) adapts to density features
3. **Runtime adaptive** - Adjust during iterations based on convergence
4. **Learning-based** - ML model predicts optimal parameters
5. **Joint optimization** - Bayesian optimization of multiple parameters

### **Recommendation**

**Start with Phase 1** (heuristic adaptive sigma):
- ✅ Immediate impact
- ✅ Low implementation cost
- ✅ No breaking changes
- ✅ Foundation for advanced methods

### **Next Steps**

1. Implement `compute_adaptive_sigma()` function
2. Add to `mfg_pde/utils/adaptive_parameters.py`
3. Create examples showing usage
4. Document in user guide
5. Gather feedback before Phase 2

---

**Last Updated**: 2025-11-03
**Status**: Design proposal awaiting implementation
**Priority**: High (significant usability improvement)
