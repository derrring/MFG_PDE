# MFGComponents as Environment Configuration

**Date**: 2025-11-03
**Insight**: MFGComponents = Environment configuration for MFGProblem
**Credit**: User observation

---

## The Key Insight

**MFGComponents is the environment configuration of MFGProblem**

Just like:
- A physics simulation has environment parameters (gravity, friction, air resistance)
- A reinforcement learning problem has environment dynamics
- An optimization problem has constraints and objective function

**MFGProblem has MFGComponents** that configure the environment where agents operate.

---

## What MFGComponents Configures

### **1. Physics of the Environment**

```python
components = MFGComponents(
    # How agents interact with the world
    hamiltonian_func=...,          # Cost of movement (kinetic energy)
    potential_func=...,            # External forces/obstacles

    # Interaction physics
    coupling_func=...,             # How agents affect each other
    node_interaction_func=...,     # Interactions at locations

    # Noise/uncertainty
    noise_intensity=0.1,           # Diffusion coefficient
    common_noise_func=...,         # Shared environmental uncertainty
)
```

### **2. Boundary Conditions (Environment Constraints)**

```python
components = MFGComponents(
    boundary_conditions=BoundaryConditions(
        type="neumann",      # How environment edges behave
        left_value=0.0,      # Wall? Exit? Periodic?
        right_value=0.0
    )
)
```

### **3. Initial State of the World**

```python
components = MFGComponents(
    initial_density_func=...,      # Where are agents initially?
    final_value_func=...,          # Terminal rewards/costs
)
```

### **4. Geometry/Topology of the Environment**

```python
components = MFGComponents(
    network_geometry=...,          # Discrete locations (graph)
    # OR continuous domain is specified in MFGProblem
)
```

---

## Analogy: Video Game Environment

Think of MFGProblem like a game level:

| Game Concept | MFG Equivalent | Where Specified |
|:-------------|:---------------|:----------------|
| Map terrain | Potential function V(x,t) | MFGComponents.potential_func |
| Movement physics | Hamiltonian H(x,m,p,t) | MFGComponents.hamiltonian_func |
| Obstacle collisions | Boundary conditions | MFGComponents.boundary_conditions |
| Starting positions | Initial density m₀(x) | MFGComponents.initial_density_func |
| Win conditions | Terminal value g(x) | MFGComponents.final_value_func |
| Player interactions | Coupling terms | MFGComponents.coupling_func |
| Map size/shape | Domain [xmin, xmax] | MFGProblem constructor |
| Time duration | Time horizon T | MFGProblem constructor |

**MFGProblem** = The complete game level (environment + rules)

**MFGComponents** = The environment configuration

---

## Three Layers of Configuration

```
┌─────────────────────────────────────────────┐
│ MFGProblem (The Complete Problem)          │
│ - Domain size (xmin, xmax, Nx)             │
│ - Time horizon (T, Nt)                     │
│ - Default parameters (sigma, coupling_coefficient)       │
└─────────────────────┬───────────────────────┘
                      │ uses
                      ▼
┌─────────────────────────────────────────────┐
│ MFGComponents (Environment Config)         │
│ - Physics (Hamiltonian, potential)         │
│ - Interactions (coupling)                  │
│ - Constraints (boundary conditions)        │
│ - Initial/terminal conditions              │
└─────────────────────┬───────────────────────┘
                      │ defines what
                      ▼
┌─────────────────────────────────────────────┐
│ Modular Solvers (How to Solve)            │
│ - HJB solver choice (FDM, WENO, GFDM)     │
│ - FP solver choice (Particle, FDM)        │
│ - Coupling algorithm (Fixed point, etc.)  │
└─────────────────────────────────────────────┘
```

**Key relationships**:
- **MFGProblem**: "I have a 1D domain [0,1] for 10 seconds"
- **MFGComponents**: "The environment has quadratic movement cost, congestion penalty, and exits at the boundaries"
- **Modular Solvers**: "I'll solve the HJB with WENO and FP with particles"

---

## Why This Separation Matters

### **1. Reusability**

Same environment, different domains:
```python
# Define environment physics once
crowd_environment = MFGComponents(
    hamiltonian_func=crowd_hamiltonian,
    boundary_conditions=exit_boundaries
)

# Use in different-sized domains
small_room = MFGProblem(xmin=0, xmax=10, Nx=50, components=crowd_environment)
large_hall = MFGProblem(xmin=0, xmax=100, Nx=200, components=crowd_environment)
```

### **2. Composability**

Mix and match environment features:
```python
# Environment pieces
epidemic_physics = MFGComponents(hamiltonian_func=epidemic_H)
mobility_restrictions = MFGComponents(boundary_conditions=quarantine_bc)

# Combine (future: support merging)
combined = MFGComponents(
    hamiltonian_func=epidemic_H,
    boundary_conditions=quarantine_bc
)
```

### **3. Separation of Concerns**

```python
# Domain expert defines environment
physics_components = MFGComponents(
    hamiltonian_func=domain_specific_H,
    potential_func=domain_specific_V
)

# Numerical analyst chooses algorithms
problem = MFGProblem(Nx=100, Nt=50, components=physics_components)
hjb = HJBWENOSolver(problem)  # Numerical choice
fp = FPParticleSolver(problem, num_particles=10000)  # Numerical choice
```

### **4. Domain Templates Can Provide Environment Presets**

```python
# Template provides environment configuration
def create_crowd_motion_solver(domain_size, ...):
    # Domain-specific environment
    crowd_environment = MFGComponents(
        hamiltonian_func=crowd_hamiltonian,
        potential_func=exit_attraction,
        boundary_conditions=wall_boundaries
    )

    # Create problem with environment
    problem = MFGProblem(
        xmin=0, xmax=domain_size,
        Nx=100, Nt=50,
        components=crowd_environment  # Environment config
    )

    # Compose solver (numerical methods)
    hjb = HJBFDMSolver(problem)
    fp = FPParticleSolver(problem, num_particles=5000)
    return FixedPointIterator(problem, hjb, fp)
```

---

## Comparison to Other Frameworks

### **OpenAI Gym / Gymnasium**

```python
# Gym: Environment configuration
env = gym.make('CartPole-v1', **env_kwargs)

# MFG_PDE: Environment configuration
components = MFGComponents(hamiltonian_func=..., potential_func=...)
problem = MFGProblem(Nx=50, Nt=20, components=components)
```

**Similarity**: Both separate environment definition from algorithm choice

### **PyTorch Neural Networks**

```python
# PyTorch: Model configuration
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

# MFG_PDE: Environment configuration
components = MFGComponents(
    hamiltonian_func=quadratic_H,
    coupling_func=interaction_term
)
```

**Similarity**: Both use composition to build complex systems from simple pieces

---

## Updated Terminology

### **Old Understanding**
- MFGComponents = "Custom problem definition"
- Implies only for custom/advanced use

### **New Understanding**
- MFGComponents = "Environment configuration"
- Universal concept that applies to all problems
- Default problems just use implicit default environment configuration

### **Implications**

```python
# This problem...
problem = MFGProblem(Nx=50, Nt=20, T=1.0)

# ...implicitly uses default environment configuration:
implicit_components = MFGComponents(
    hamiltonian_func=lambda x, m, p, t: 0.5 * p**2 + problem.coupling_coefficient * m,
    potential_func=lambda x, t: default_potential(x),
    initial_density_func=lambda x: default_gaussian(x),
    boundary_conditions=default_bc
)

# Making it explicit:
problem = MFGProblem(Nx=50, Nt=20, T=1.0, components=explicit_components)
```

---

## Design Implications

### **1. Documentation Should Emphasize "Environment"**

**Old**: "For custom problems, provide MFGComponents"
**New**: "Configure your environment with MFGComponents"

### **2. Examples Should Show Environment Configuration**

```python
# Good example structure:
# 1. Define environment
environment = MFGComponents(
    hamiltonian_func=my_physics,
    potential_func=my_terrain
)

# 2. Create problem with environment
problem = MFGProblem(
    xmin=0, xmax=1, Nx=50,
    T=1.0, Nt=20,
    components=environment  # Explicit environment
)

# 3. Choose numerical methods
hjb = HJBGFDMSolver(problem)
fp = FPParticleSolver(problem)
solver = FixedPointIterator(problem, hjb, fp)
```

### **3. Future: Environment Library**

Could create library of environment configurations:

```python
# mfg_pde/environments/crowd_motion.py (FUTURE)

def crowd_motion_environment(congestion_level="medium"):
    """Pre-configured crowd motion environment."""
    return MFGComponents(
        hamiltonian_func=...,
        potential_func=...,
        boundary_conditions=...
    )

# Usage:
from mfg_pde.environments import crowd_motion_environment

environment = crowd_motion_environment(congestion_level="high")
problem = MFGProblem(xmin=0, xmax=100, Nx=200, components=environment)
```

---

## Summary

**MFGComponents = Environment Configuration**

| Concept | Configures | Example |
|:--------|:-----------|:--------|
| **Environment Physics** | How agents move and interact | Hamiltonian, coupling |
| **Environment Constraints** | Boundaries and obstacles | Boundary conditions |
| **Environment State** | Initial and terminal conditions | m₀(x), g(x) |
| **Environment Topology** | Spatial structure | Network geometry |

**This framing**:
- ✅ Clarifies purpose of MFGComponents
- ✅ Shows relationship to MFGProblem (problem = domain + environment)
- ✅ Distinguishes from modular solvers (algorithms)
- ✅ Provides mental model for users
- ✅ Aligns with other frameworks (Gym, PyTorch)
- ✅ Opens path for environment libraries

**Key insight**: Every MFG problem has an environment. MFGComponents makes it explicit and configurable.

---

**Last Updated**: 2025-11-03
**Status**: Conceptual clarification - documentation should adopt this framing
**Credit**: User insight about environment configuration
