# Algorithm Structure Reorganization Plan

**Status**: [WIP] Planning Phase
**Date**: 2025-09-29
**Scope**: Complete restructuring of `mfg_pde/alg/` directory for improved conceptual clarity

## Executive Summary

The current `mfg_pde/alg/` structure mixes mathematical approaches (variational vs numerical) with equation types (HJB vs FP), creating conceptual inconsistency. This document outlines a reorganization plan that groups algorithms by mathematical paradigm while maintaining backward compatibility.

## Current Structure Analysis

### Existing Organization
```
mfg_pde/alg/
├── hjb_solvers/         # Individual HJB equation solvers
├── fp_solvers/          # Individual FP equation solvers
├── mfg_solvers/         # Coupled system solvers
├── variational_solvers/ # Direct optimization methods
└── neural_solvers/      # PINN-based approaches
```

### Issues with Current Structure

1. **Conceptual Inconsistency**: Mixes equation types (`hjb_solvers/`, `fp_solvers/`) with mathematical approaches (`variational_solvers/`, `neural_solvers/`)

2. **Unclear Boundaries**: Where does a coupled numerical method belong? In `mfg_solvers/` or based on its technique?

3. **Growth Limitations**: No clear place for reinforcement learning, quantum computing, or other emerging paradigms

4. **User Confusion**: Researchers familiar with specific approaches may struggle to find relevant solvers

## Proposed New Structure

### Organizational Principle
**Group by mathematical paradigm, not by equation type**

```
mfg_pde/alg/
├── numerical/           # Classical numerical analysis methods
│   ├── hjb_solvers/     # Finite difference, WENO, semi-Lagrangian
│   ├── fp_solvers/      # FDM, particle methods, spectral
│   └── mfg_solvers/     # Coupled numerical schemes
├── optimization/        # Direct optimization approaches
│   ├── variational_methods/     # Direct functional minimization
│   ├── optimal_transport/       # Wasserstein, JKO schemes
│   ├── augmented_lagrangian/    # Constraint handling
│   └── primal_dual/            # Saddle point methods
├── neural/              # Neural network-based methods
│   ├── core/            # Shared neural infrastructure and architectures
│   ├── physics_informed/# Physics-informed neural networks
│   └── operator_learning/ # Neural operator and data-driven approaches
└── reinforcement/       # Reinforcement learning paradigm
    ├── core/            # Shared RL infrastructure and base classes
    ├── algorithms/      # Specific RL algorithms adapted for MFG
    └── approaches/      # Mathematical approaches (value/policy/actor-critic)
```

### Benefits of New Structure

1. **Conceptual Clarity**: Each top-level directory represents a distinct mathematical approach
2. **Natural Navigation**: Researchers can easily find methods from their domain
3. **Extensibility**: Clear placement for future paradigms (quantum, stochastic, etc.)
4. **Cross-Pollination**: Easier to identify hybrid opportunities between paradigms
5. **Educational Value**: Structure reflects the mathematical taxonomy of the field

## Detailed Directory Design

### `numerical/` - Classical Numerical Analysis
**Philosophy**: Discretization-based methods with convergence analysis

- **`hjb_solvers/`**: Individual HJB equation solvers
  - Finite difference schemes
  - WENO methods
  - Semi-Lagrangian approaches
  - Spectral methods

- **`fp_solvers/`**: Individual Fokker-Planck solvers
  - Finite difference methods
  - Particle-based methods
  - Finite element approaches
  - Monte Carlo methods

- **`mfg_solvers/`**: Coupled system numerical methods
  - Fixed-point iterations
  - Newton-type methods
  - Splitting schemes
  - Adaptive mesh refinement

### `optimization/` - Direct Optimization Approaches
**Philosophy**: Minimize functionals directly without discretizing PDEs

**Refined Organization**: Based on algorithmic families rather than mathematical theory boundaries, since variational calculus underlies most optimization methods.

- **`variational_methods/`**: Direct functional minimization
  - Direct optimization of MFG functionals without PDE discretization
  - Gradient descent on infinite-dimensional spaces
  - Riemannian optimization methods

- **`optimal_transport/`**: Wasserstein and transport-based methods
  - JKO (Jordan-Kinderlehrer-Otto) schemes
  - Wasserstein gradient flows
  - Optimal transport formulations of MFG

- **`augmented_lagrangian/`**: Constraint handling methods
  - Augmented Lagrangian for mass conservation
  - Penalty methods for boundary conditions
  - ADMM (Alternating Direction Method of Multipliers)

- **`primal_dual/`**: Saddle point optimization
  - Primal-dual algorithms for min-max problems
  - Chambolle-Pock schemes
  - Arrow-Hurwicz methods

### `neural/` - Neural Network Methods
**Philosophy**: Function approximation with neural networks and interconnected learning approaches

**Key Insight**: Neural methods for PDEs share common foundations (architectures, training, optimization) but differ in loss function design. Most approaches can be combined in hybrid methods that leverage both physics constraints and data-driven learning.

- **`core/`**: Shared neural infrastructure
  - Common network architectures (MLPs, CNNs, Transformers for MFG)
  - Shared training loops and optimization strategies
  - Base loss function classes and regularization methods
  - Neural utilities (initialization, activation functions)

- **`physics_informed/`**: Physics-informed neural networks
  - `classic_pinn.py` - Standard PINNs for individual PDEs
  - `mfg_pinn.py` - Coupled PINN for complete MFG systems
  - `adaptive_pinn.py` - Adaptive sampling and loss weighting
  - `multi_scale_pinn.py` - Multi-scale physics constraints

- **`operator_learning/`**: Neural operator and data-driven approaches
  - `neural_operators.py` - DeepONet, FNO, Neural ODEs for MFG
  - `transformer_pde.py` - Transformer-based PDE solvers
  - `hybrid_methods.py` - Physics + data-driven combinations
  - `surrogate_models.py` - Fast neural surrogates for real-time MFG

#### Neural Method Interconnections

This structure recognizes that **neural approaches are highly interconnected**:

1. **Shared Architectures**: Same MLP can be used with physics loss (PINN) or data loss (operator learning)
2. **Hybrid Training**: Combine physics constraints with data-driven learning
3. **Transfer Learning**: Pre-train on data, fine-tune with physics constraints
4. **Multi-Task Learning**: Simultaneously learn from multiple loss sources
5. **Progressive Learning**: Start with simple physics, add complexity and data

**Example Cross-Connections**:
- Hybrid PINN uses physics loss + operator learning for data integration
- Transfer learning: pre-train neural operator on data, fine-tune with physics
- Multi-scale methods combine local PINN with global operator learning

### `reinforcement/` - Reinforcement Learning Paradigm
**Philosophy**: Multi-agent learning and equilibrium computation with interconnected methods

**Key Insight**: Mean Field RL serves as an **integration hub** connecting finite-population multi-agent methods with classical MFG theory. Most RL algorithms can be adapted to mean field settings through population scaling.

- **`core/`**: Shared RL infrastructure
  - Base agent classes for MFG environments
  - MFG environment wrappers and interfaces
  - Common training loops and evaluation metrics
  - Population state representations

- **`algorithms/`**: Specific RL algorithms adapted for MFG
  - `mfrl.py` - Mean Field RL (Yang et al.) as central algorithm
  - `nash_q.py` - Nash Q-learning for finite populations
  - `maddpg.py` - Multi-Agent DDPG with mean field approximation
  - `population_ppo.py` - PPO adapted for population dynamics
  - `mean_field_ac.py` - Actor-critic with population state

- **`approaches/`**: Mathematical approaches applicable across algorithms
  - `value_based/` - Q-learning family (DQN, mean field Q-learning)
  - `policy_based/` - Policy gradient family (REINFORCE, PPO variants)
  - `actor_critic/` - AC family (A3C, SAC adaptations for MFG)

#### Interconnection Philosophy

This structure recognizes that **Mean Field RL is not standalone** but rather:

1. **Population Scaling Bridge**: Multi-agent methods → Mean field as N→∞
2. **Algorithm Adaptation**: Most RL algorithms have mean field variants
3. **Mathematical Unification**: Same underlying principles (Bellman equations, policy gradients) applied to population settings
4. **Cross-Method Learning**: Techniques developed in one approach benefit others

**Example Cross-Connections**:
- Mean field Q-learning uses value-based approach with population state
- Population PPO combines policy gradients with mean field state representation
- Nash-Q learning provides finite-population approximation to mean field equilibrium

## Migration Strategy

### Phase 1: Prepare New Structure (No Breaking Changes)
1. Create new directory structure alongside existing
2. Copy files to new locations with updated imports
3. Create comprehensive aliases in `__init__.py` files
4. Update documentation to reference new structure

### Phase 2: Deprecation Warnings
1. Add deprecation warnings to old import paths
2. Update examples to use new structure
3. Provide migration guide for users
4. Test backward compatibility extensively

### Phase 3: Complete Migration
1. Remove old directory structure
2. Clean up alias imports
3. Update all internal references
4. Finalize documentation

### Backward Compatibility Strategy

Maintain compatibility through smart `__init__.py` files:

```python
# mfg_pde/alg/__init__.py
import warnings
from .numerical.hjb_solvers import *
from .numerical.fp_solvers import *
from .optimization.variational import *
from .neural.pinn import *

# Legacy imports with deprecation warnings
def __getattr__(name):
    if name in ['hjb_solvers', 'fp_solvers', 'variational_solvers', 'neural_solvers']:
        warnings.warn(
            f"Importing {name} from mfg_pde.alg is deprecated. "
            f"Use the new structure: mfg_pde.alg.{new_path_mapping[name]}",
            DeprecationWarning,
            stacklevel=2
        )
        return getattr(import_new_location(name), name)
    raise AttributeError(f"module 'mfg_pde.alg' has no attribute '{name}'")
```

## Reinforcement Learning Integration

### Mathematical Connections
- **HJB equation ↔ Bellman equation**: Direct correspondence for value functions
- **Optimal control ↔ Policy optimization**: Natural mapping between paradigms
- **Nash equilibrium ↔ Multi-agent equilibrium**: Same mathematical concept
- **Population dynamics ↔ Multi-agent learning**: Core equivalence in large populations

### Implementation Approach

**Revised Base Class Hierarchy** (showing neural integration):
```python
# mfg_pde/alg/reinforcement/core/base_rl.py
class BaseRLSolver(BaseMFGSolver):
    """Base class for RL-based MFG solvers with interconnected design"""

    def __init__(self, environment, agent_config, population_config):
        super().__init__()
        self.env = environment
        self.population_size = population_config.get('size', float('inf'))  # finite vs mean field
        self.agents = self._create_agents(agent_config)

    def train(self) -> RLSolverResult:
        """Train agents with population-aware learning"""
        pass

    def evaluate_nash_gap(self) -> float:
        """Compute Nash equilibrium gap (works for finite and mean field)"""
        pass

    def scale_to_mean_field(self):
        """Convert finite-population solution to mean field limit"""
        pass

# mfg_pde/alg/reinforcement/core/environments.py
class MFGEnvironment:
    """Environment wrapper that handles both finite and mean field populations"""

    def __init__(self, mfg_problem, population_type='mean_field'):
        self.problem = mfg_problem
        self.population_type = population_type  # 'finite' or 'mean_field'
```

**Example Algorithm Integration** (showing interconnections):
```python
# mfg_pde/alg/reinforcement/algorithms/mfrl.py
from ..approaches.policy_based import PolicyGradientBase
from ..core import MFGEnvironment

class MeanFieldRLSolver(PolicyGradientBase):
    """Yang et al. MFRL - integrates policy gradients with mean field theory"""

# mfg_pde/alg/reinforcement/algorithms/nash_q.py
from ..approaches.value_based import ValueBasedBase

class NashQSolver(ValueBasedBase):
    """Finite population that approximates mean field equilibrium"""
```

### Research Integration Opportunities

1. **Cross-Paradigm Hybrid Methods**:
   - **Neural-RL**: Use neural networks as function approximators in RL (natural fit)
   - **Numerical-RL**: RL-guided adaptive mesh refinement and solver selection
   - **Optimization-RL**: RL for hyperparameter tuning of variational methods

2. **Multi-Method Benchmarking**:
   - Compare numerical vs neural vs RL convergence rates
   - Analyze computational complexity across paradigms
   - Identify problem classes where each approach excels

3. **Hybrid Neural-RL Architecture**:
   ```python
   # Example: Neural networks within RL agents
   from ..neural.core import MFGNetwork
   from ..reinforcement.algorithms import MeanFieldRLSolver

   class NeuralMFRLSolver(MeanFieldRLSolver):
       """RL solver using neural networks for value/policy approximation"""
       def __init__(self, mfg_problem):
           # Use shared neural architectures
           self.value_net = MFGNetwork(input_dim=mfg_problem.state_dim)
           self.policy_net = MFGNetwork(input_dim=mfg_problem.state_dim)
   ```

4. **Progressive Learning Pipeline**:
   - **Stage 1**: Numerical solution for initialization
   - **Stage 2**: Neural network learning from numerical data
   - **Stage 3**: RL fine-tuning for robustness and adaptation

## Configuration Management Strategy

### Multi-Paradigm Configuration System
**Challenge**: Different paradigms have radically different configuration needs:
- **Numerical**: Grid parameters, convergence tolerances, solver types
- **Neural**: Network architectures, training hyperparameters, optimization settings
- **RL**: Environment configs, agent parameters, training schedules
- **Optimization**: Algorithm choices, constraint handling, step sizes

### Proposed Solution: Hydra-Based Hierarchical Configs
```
configs/
├── paradigm/           # Paradigm-specific base configs
│   ├── numerical.yaml
│   ├── neural.yaml
│   ├── reinforcement.yaml
│   └── optimization.yaml
├── solvers/           # Specific solver configurations
│   ├── numerical/
│   │   ├── hjb_fdm.yaml
│   │   └── mfg_fixed_point.yaml
│   ├── neural/
│   │   ├── mfg_pinn.yaml
│   │   └── neural_operator.yaml
│   ├── reinforcement/
│   │   ├── mfrl.yaml
│   │   └── nash_q.yaml
│   └── optimization/
│       ├── variational_direct.yaml
│       └── optimal_transport.yaml
├── problems/          # MFG problem configurations
│   ├── crowd_motion.yaml
│   ├── financial_systemic_risk.yaml
│   └── traffic_flow.yaml
└── experiments/       # Full experiment configurations
    ├── benchmark_comparison.yaml
    └── convergence_analysis.yaml
```

### Dependency Management Strategy
```python
# setup.py extras_require
extras_require = {
    "numerical": ["scipy>=1.9.0", "scikit-sparse"],
    "neural": ["torch>=1.12.0", "optax", "jax"],
    "reinforcement": ["gymnasium", "stable-baselines3", "tensorboard"],
    "optimization": ["cvxpy", "mosek"],
    "plotting": ["plotly>=5.0", "matplotlib>=3.5"],
    "all": ["scipy>=1.9.0", "torch>=1.12.0", "gymnasium", "cvxpy", "plotly>=5.0"]
}
```

**Usage**:
```bash
# Install only what you need
pip install mfg_pde[numerical]      # Classical methods only
pip install mfg_pde[neural,rl]      # ML approaches
pip install mfg_pde[all]           # Everything
```

## Implementation Timeline

### Phase 1: Foundation (Weeks 1-3)
- **Week 1**: Refined directory structure + configuration system design
- **Week 2**: Dependency management setup + base class hierarchies
- **Week 3**: Backward compatibility layer + migration tools

### Phase 2: Core Migration (Weeks 4-6)
- **Week 4**: Migrate numerical solvers to new structure
- **Week 5**: Migrate and enhance neural solvers with interconnections
- **Week 6**: Comprehensive testing of migrated components

### Phase 3: New Paradigms (Weeks 7-9)
- **Week 7**: Implement RL base classes and first algorithms
- **Week 8**: Enhanced optimization structure with new methods
- **Week 9**: Cross-paradigm integration and hybrid methods

### Phase 4: Integration & Documentation (Weeks 10-12)
- **Week 10**: Full configuration system integration
- **Week 11**: Update all examples, tutorials, and documentation
- **Week 12**: Performance benchmarking and final testing

## Quality Assurance

### Testing Strategy
- All existing tests must pass with new structure
- Import compatibility tests for old paths
- Performance benchmarks to ensure no regression
- Documentation build verification

### Success Metrics
- Zero breaking changes for existing users
- Clear performance improvement in solver discoverability
- Successful integration of first RL solver
- Positive feedback from test users

## Future Extensions

### Potential New Paradigms
- **`quantum/`**: Quantum computing approaches to MFG
- **`stochastic/`**: Advanced stochastic analysis methods
- **`geometric/`**: Differential geometry and manifold methods
- **`probabilistic/`**: Bayesian and uncertainty quantification

### Neural Method Future Convergence
**Important Consideration**: The `physics_informed/` vs `operator_learning/` boundaries are rapidly blurring in practice. Most state-of-the-art neural PDE solvers are becoming hybrid methods that combine:

- **Physics constraints** for mathematical rigor and data efficiency
- **Data-driven learning** for handling complex, unknown dynamics
- **Multi-scale approaches** that adapt between local physics and global patterns

**Future-Proofing Strategy**:
- Keep current structure for clarity during migration
- Plan for potential `hybrid/` subdirectory as methods converge
- Ensure `core/` infrastructure supports flexible loss combination
- Design configuration system to handle multi-loss training seamlessly

### Cross-Paradigm Opportunities
- **Neural-Numerical Hybrids**: Neural-guided adaptive mesh refinement
- **RL-Optimization**: RL for hyperparameter tuning of optimization methods
- **Physics-Data Integration**: Neural networks as physics-informed function approximators

---

**Next Steps**:
1. Review and approve this reorganization plan
2. Create detailed implementation specifications
3. Begin Phase 1 migration with backward compatibility
4. Design and implement first RL solver prototypes

**References**:
- Yang, J. et al. "Mean Field Multi-Agent Reinforcement Learning" (ICML 2018)
- Ruthotto, L. & Haber, E. "Deep Neural Networks motivated by PDEs" (2019)
- E, W. & Yu, B. "The Deep Ritz Method" (2018)
