# Domain Template Design

**Date**: 2025-11-03
**Status**: Design stage - placeholders only, no implementation yet
**Philosophy**: Domain templates encode domain-specific best practices, not arbitrary quality levels

---

## Design Principles

### **What Domain Templates Should Provide**

1. **Domain Knowledge**: Hamiltonians, boundary conditions, parameters suited to specific applications
2. **Best Practices**: Validated configurations from domain literature
3. **Sensible Defaults**: Grid resolution, time steps, solver choices appropriate for domain
4. **Domain Terminology**: Options named for domain concepts, not algorithmic details

### **What Domain Templates Should NOT Be**

1. ❌ **Not generic quality levels** ("fast", "accurate" - these are meaningless without context)
2. ❌ **Not hardcoded solver combinations** (users can compose modularly if needed)
3. ❌ **Not black boxes** (templates should be inspectable and modifiable)

---

## Template Interface Design

### **Standard Signature**

```python
def create_[domain]_solver(
    problem: MFGProblem | [DomainSpecificProblemClass],
    *,
    # Domain-specific configuration
    **domain_options
) -> FixedPointIterator:
    """
    Create solver with domain-specific best practices.

    Args:
        problem: MFG problem instance (generic or domain-specific)
        **domain_options: Domain-specific configuration

    Returns:
        Composed MFG solver with domain knowledge baked in

    Note:
        This is a convenience template. Users can always use modular
        approach for full control and customization.
    """
    pass
```

### **Key Design Features**

1. **Returns composed solver** - Uses modular approach internally
2. **Accepts MFGProblem or domain class** - Flexible input
3. **Domain-specific options** - Named for domain concepts
4. **Transparent** - Implementation visible, not magic
5. **Optional** - Modular approach always available

---

## Placeholder Templates

### **Future Implementation Candidates**

#### 1. Crowd Motion / Pedestrian Dynamics

```python
# mfg_pde/templates/crowd_motion.py (FUTURE)

def create_crowd_motion_solver(
    problem: MFGProblem,
    *,
    num_particles: int = 5000,
    congestion_penalty: float = 10.0,
    use_anisotropy: bool = False,
    exit_absorption: bool = True,
    max_density: float | None = None
) -> FixedPointIterator:
    """
    Solver for crowd evacuation and pedestrian dynamics.

    Domain knowledge:
    - High congestion penalty (typical: 5-15)
    - Particle methods for mass conservation
    - Anisotropic diffusion (optional)
    - Exit boundary conditions

    Literature:
    - Hughes (2002): Social force model
    - Degond et al. (2013): MFG for pedestrian flow
    """
    # TODO: Implement when mature use cases emerge
    raise NotImplementedError("Crowd motion template - future implementation")
```

#### 2. Epidemic Models

```python
# mfg_pde/templates/epidemic.py (FUTURE)

def create_epidemic_solver(
    problem: MFGProblem,
    *,
    transmission_rate: float = 0.3,
    recovery_rate: float = 0.1,
    mobility_cost: float = 1.0,
    quarantine_regions: list[tuple[float, float]] | None = None
) -> FixedPointIterator:
    """
    Solver for epidemic models (SIR/SEIR dynamics).

    Domain knowledge:
    - Network or spatial formulation
    - Transmission proportional to density
    - Mobility vs. infection tradeoff

    Literature:
    - Laguzet & Turinici (2015): MFG epidemic control
    - Achdou et al. (2020): COVID-19 MFG model
    """
    # TODO: Implement when mature use cases emerge
    raise NotImplementedError("Epidemic template - future implementation")
```

#### 3. Finance / Portfolio Optimization

```python
# mfg_pde/templates/finance.py (FUTURE)

def create_finance_solver(
    problem: MFGProblem,
    *,
    risk_aversion: float = 0.5,
    transaction_cost: float = 0.01,
    market_impact: bool = True,
    volatility: float = 0.2
) -> FixedPointIterator:
    """
    Solver for portfolio optimization and trading games.

    Domain knowledge:
    - Mean-variance preferences
    - Transaction costs
    - Market impact from density
    - Stochastic volatility

    Literature:
    - Lasry & Lions (2007): Finance applications
    - Cardaliaguet & Lehalle (2018): Market microstructure
    """
    # TODO: Implement when mature use cases emerge
    raise NotImplementedError("Finance template - future implementation")
```

#### 4. Traffic Flow

```python
# mfg_pde/templates/traffic.py (FUTURE)

def create_traffic_solver(
    problem: MFGProblem,
    *,
    free_flow_speed: float = 1.0,
    congestion_coefficient: float = 2.0,
    max_density: float = 1.0,
    toll_regions: dict[tuple[float, float], float] | None = None
) -> FixedPointIterator:
    """
    Solver for traffic flow and congestion pricing.

    Domain knowledge:
    - Speed-density relationship
    - Congestion quadratic in density
    - Toll/pricing mechanisms
    - Capacity constraints

    Literature:
    - Achdou & Camilli (2013): Traffic congestion
    - Bauso et al. (2016): Congestion games
    """
    # TODO: Implement when mature use cases emerge
    raise NotImplementedError("Traffic template - future implementation")
```

---

## Implementation Strategy

### **Phase 1: Design & Documentation (CURRENT)**
- ✅ Define interface standards
- ✅ Document design principles
- ✅ Create placeholder signatures
- ✅ Specify what templates should/shouldn't be

### **Phase 2: Validation (FUTURE)**
- Identify 2-3 mature use cases per domain
- Validate configurations from literature
- Test on canonical problems
- Document domain-specific considerations

### **Phase 3: Implementation (FUTURE)**
- Implement validated templates
- Add domain-specific problem classes (optional)
- Create domain-specific examples
- Write domain-focused documentation

### **Phase 4: Maintenance (FUTURE)**
- Update as domain research advances
- Add new domains as use cases mature
- Deprecate templates if better patterns emerge

---

## Decision Criteria: When to Implement a Template

**Implement when**:
✅ **Multiple validated use cases** (≥3 different problems in literature)
✅ **Consensus on best practices** (established parameter ranges)
✅ **User demand** (repeated questions about how to configure for domain)
✅ **Stable Hamiltonian form** (domain physics well-understood)

**Don't implement if**:
❌ Only 1-2 examples exist
❌ Active research area (no consensus yet)
❌ Domain-specific physics unclear
❌ Users satisfied with modular approach

---

## Relationship to Modular Approach

**Templates are NOT a replacement for modular approach**:

```python
# Modular approach (ALWAYS available, RECOMMENDED)
hjb = HJBGFDMSolver(problem)
fp = FPParticleSolver(problem, num_particles=10000)
solver = FixedPointIterator(problem, hjb, fp, damping_factor=0.7)

# Domain template (OPTIONAL convenience, FUTURE)
solver = create_crowd_motion_solver(problem, num_particles=10000)

# Both do the same thing! Template uses modular approach internally.
# Template just encodes domain knowledge about good defaults.
```

**Key insight**: Templates are **convenience wrappers** around modular composition, not a separate API level.

---

## Current Status

**No implementation yet** - only design and placeholders.

**Why wait**:
1. Need mature use cases to validate configurations
2. Domain best practices still emerging in some areas
3. Users can use modular approach for all current needs
4. Premature abstraction risk (building wrong interface)

**When to revisit**:
- After 3+ papers using MFG_PDE for same domain
- When users repeatedly ask "how do I set up [domain]?"
- When domain-specific Hamiltonians become standardized

---

**Last Updated**: 2025-11-03
**Status**: Design stage - implementation deferred until validated use cases emerge
