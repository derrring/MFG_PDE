# Design Session Summary - 2025-11-03

**Topics**: API design refinement, Hamiltonian architecture, modular vs factory patterns
**Status**: Design decisions finalized, implementation deferred

---

## Key Decisions

### **1. Eliminated "Solver Tiers" Terminology** ✅

**Rationale**: "Tier 1/2/3" conflates method choice with arbitrary quality levels

**Changes Made**:
- Updated `PROGRESSIVE_DISCLOSURE_API_DESIGN.md` - Removed all tier references
- Updated `TWO_LEVEL_API_INTEGRITY_CHECK.md` - Replaced "Solver Tiers" with "Factory Functions"
- Created `SOLVER_TIER_ELIMINATION_SUMMARY.md` - Documents changes and remaining work

**New Philosophy**:
- **Method-based descriptions**: FDM-only, hybrid, WENO, etc.
- **Quality depends on configuration**: Grid resolution, tolerance, damping - not function names
- **No arbitrary "tiers"**: Each method has strengths for different problems

### **2. Prioritized Modular Approach Over Generic Factories** ✅

**User Insight**: "I'd prefer to create solver by myself. Otherwise they are also hard coded in function name"

**Key Realization**: Names like `create_fast_solver()` and `create_accurate_solver()` are just as arbitrary as "Tier 1/2/3"

**New Hierarchy**:
1. **Modular Approach** (Recommended)
   - Explicit composition: `HJBSolver + FPSolver + FixedPointIterator`
   - Clear what you're getting
   - Full control over all parameters

2. **Domain Templates** (Future)
   - Domain-specific patterns: `create_crowd_motion_solver()`, `create_epidemic_solver()`
   - Encode domain knowledge, not arbitrary quality levels
   - Example: crowd motion → high congestion penalty, particle methods for mass conservation

3. **Generic Factories** (Legacy)
   - Exist but not recommended
   - Lack domain knowledge
   - Names are context-dependent

**Documentation Updated**:
- Executive summary now emphasizes modular approach
- Added "Legacy: Generic Factory Functions (Not Recommended)" section
- All examples show modular composition
- Updated summary tables and quick references

### **3. Domain Templates - Design Only, No Implementation** ✅

**Decision**: Not ready to implement yet

**Reasons**:
1. Need validated use cases (≥3 papers/applications per domain)
2. Domain best practices still emerging
3. Risk of premature abstraction
4. Users satisfied with modular approach

**Created**: `DOMAIN_TEMPLATE_DESIGN.md`
- Interface specifications
- Design principles
- Placeholder signatures (crowd, epidemic, finance, traffic)
- Implementation criteria

**When to implement**:
- After 3+ validated use cases in literature
- When users repeatedly ask "how do I set up [domain]?"
- When domain Hamiltonians become standardized

### **4. Unified Hamiltonian Abstraction** 📝 PROPOSED

**Motivation**: Three different Hamiltonian representations lack unified interface

**Current State**:
- Problem-level method (`MFGProblem.H()`)
- Custom functions (`MFGComponents.hamiltonian_func`)
- Default implementation (built-in quadratic)

**Proposed**: `BaseHamiltonian` abstract class

**Created**: `HAMILTONIAN_ABSTRACTION_DESIGN.md`

**Key Features**:
```python
class BaseHamiltonian(ABC):
    @abstractmethod
    def evaluate(self, x, m, p, t=0.0) -> float:
        pass

    def derivative_m(self, x, m, p, t=0.0) -> float:
        # Default: numerical differentiation
        pass

    def derivative_p(self, x, m, p, t=0.0) -> float:
        # Default: numerical differentiation
        pass
```

**Concrete Implementations Proposed**:
- `QuadraticHamiltonian` - Default H = (1/2)p² + λm
- `PowerLawHamiltonian` - H = (1/γ)|p|^γ + λm^β
- `CallableHamiltonian` - Wrap arbitrary functions
- `CompositeHamiltonian` - Sum of multiple Hamiltonians

**Benefits**:
- Composability (build from simple pieces)
- Reusability (define once, use everywhere)
- Type safety (unified interface)
- Extensibility (users can subclass)

**Integration with Domain Templates**:
Templates can provide pre-configured Hamiltonian objects with domain knowledge.

**Implementation Priority**: Medium (after domain template design, before implementation)

### **5. MFGComponents vs Modular Problem** ✅ CLARIFIED

**Question**: Can MFGComponents be replaced by modular approach?

**Answer**: **No** - they serve different purposes

**Key Distinction**:
| Aspect | MFGComponents | Modular Solvers |
|:-------|:--------------|:----------------|
| **Level** | Problem (physics) | Algorithm (numerics) |
| **Defines** | Hamiltonian, BCs | HJB solver, FP solver |
| **When** | Problem creation | Solver creation |
| **Abstraction** | Mathematical formulation | Numerical methods |

**Architecture**:
```
MFGComponents (WHAT - physics)
      ↓
   MFGProblem (problem data)
      ↓
Modular Solvers (HOW - numerics)
```

**Both are valuable** and operate at different levels.

**Created**: `MFGCOMPONENTS_VS_MODULAR_ANALYSIS.md`

**Recommendation**:
- Keep MFGComponents for problem definition
- Keep modular solvers for algorithm composition
- Future: Add domain-specific problem classes (e.g., `CrowdMotionProblem`) that use MFGComponents internally

---

## Files Created

### **Design Documents**
1. `DOMAIN_TEMPLATE_DESIGN.md` - Interface specs and placeholders
2. `HAMILTONIAN_ABSTRACTION_DESIGN.md` - Unified Hamiltonian abstraction proposal
3. `MFGCOMPONENTS_VS_MODULAR_ANALYSIS.md` - Clarifies different abstraction levels
4. `MFGCOMPONENTS_AS_ENVIRONMENT.md` - Environment configuration framing
5. `DIMENSION_AGNOSTIC_COMPONENTS.md` - Dimension-agnostic design analysis
6. `MFGCOMPONENTS_COMPREHENSIVE_AUDIT.md` - Coverage audit and gap analysis
7. `MFGCOMPONENTS_VALIDATION_DESIGN.md` - Validation logic specification
8. `MFGCOMPONENTS_BUILDER_DESIGN.md` - Helper/builder functions design
9. `DESIGN_SESSION_SUMMARY_2025-11-03.md` - This file

## Files Updated

### **Design Documents**
1. `PROGRESSIVE_DISCLOSURE_API_DESIGN.md` - Major revision
   - Removed all "Solver Tier" references
   - Prioritized modular approach over factory mode
   - Added domain template examples
   - Updated all tables and examples

2. `SOLVER_TIER_ELIMINATION_SUMMARY.md` - Status update
   - Marked design docs as fully updated
   - Listed comprehensive changes

### **Core Code**
3. `mfg_pde/core/mfg_problem.py` - MFGComponents extensions
   - Updated docstring to "environment configuration" framing
   - Added 37 new optional fields across 6 categories:
     * Neural Network MFG (7 fields)
     * Reinforcement Learning MFG (7 fields)
     * Implicit Geometry (6 fields)
     * Adaptive Mesh Refinement (7 fields)
     * Time-Dependent Domains (5 fields)
     * Multi-Population MFG (5 fields)
   - All additions 100% backward compatible (default None)

---

## Implementation Roadmap

### **Immediate (No Code Changes)**
- ✅ Design documentation complete
- ✅ Philosophy clarified
- ✅ Interfaces specified

### **Short Term (Next)**
1. Update user-facing documentation
   - `docs/user/SOLVER_SELECTION_GUIDE.md` - Restructure around methods, not tiers
   - `docs/user/quickstart.md` - Emphasize modular approach
   - `docs/user/core_objects.md` - Remove "Tier 2" references

2. Add examples demonstrating modular approach
   - Method comparison using explicit composition
   - Custom Hamiltonian examples
   - Parameter sweep examples

### **Medium Term (When Ready)**
1. Implement Hamiltonian abstraction
   - Create `mfg_pde/core/hamiltonian.py`
   - Implement concrete classes (Quadratic, PowerLaw, etc.)
   - Integrate with MFGComponents (backward compatible)

2. Add deprecation warnings to generic factories
   - Guide users to modular approach
   - Provide migration examples

### **Long Term (Future)**
1. Implement domain templates (when validated)
   - Start with 1-2 mature domains
   - Provide pre-configured Hamiltonian objects
   - Create domain-specific problem classes

2. Create migration guide
   - Generic factories → Modular approach
   - Legacy Hamiltonians → Hamiltonian objects

---

## Key Principles Established

### **1. Domain Knowledge Over Arbitrary Labels**
Don't say "fast" or "accurate" - say what the method is (FDM, WENO, particle) and let users decide based on their domain.

### **2. Modular Composition as Foundation**
Explicit composition (`HJB + FP + Coupling`) gives users understanding and control. Templates are optional convenience, not a separate API level.

### **3. Quality Depends on Configuration**
Grid resolution, time steps, tolerance, damping - these determine quality, not function names or arbitrary tiers.

### **4. Design Before Implementation**
Specify interfaces and validate use cases before rushing to implement. Placeholders prevent premature abstraction.

### **5. Separation of Concerns**
- Physics (MFGComponents) vs Numerics (Modular Solvers)
- Problem Definition vs Algorithm Selection
- What to Solve vs How to Solve

---

## User Impact

### **Breaking Changes**: None
All existing code continues to work. This is purely design clarification and documentation improvement.

### **Recommended Migration**:
Users currently using generic factories (`create_fast_solver`, etc.) should consider modular approach for:
- Better understanding of what they're using
- Full control over configuration
- Easier customization and experimentation

### **Future Deprecation Timeline**:
1. **Now**: Document modular approach as recommended
2. **v0.10.0**: Add deprecation warnings to generic factories
3. **v1.0.0**: Remove generic factories (major version bump)

---

## Questions Addressed

### **Q1: When numerical method intervenes with Hamiltonian, do we have a solution?**
**A**: Yes - layered architecture with clear dispatch:
- Problem-level dispatcher (`MFGProblem.H()`)
- Custom Hamiltonians via `MFGComponents.hamiltonian_func`
- Default implementation (quadratic)
- No overlapping functions - clean separation

### **Q2: What Hamiltonians do we use?**
**A**: Three ways:
1. Default (built-in quadratic: H = 0.5|∇u|² + λm)
2. Custom via MFGComponents (user-defined functions)
3. HamiltonianAdapter for legacy signatures

### **Q3: Should we implement domain factories now?**
**A**: No - design and placeholders only. Wait for validated use cases.

### **Q4: Can we have unified abstract Hamiltonian?**
**A**: Yes - excellent idea. Proposed `BaseHamiltonian` abstraction. Implement before domain templates.

### **Q5: Can MFGComponents be replaced by modular approach?**
**A**: No - they serve different purposes. MFGComponents defines problem physics, modular approach composes numerical methods. Both valuable at different abstraction levels.

---

## Session Extension: MFGComponents Implementation

### **Completed Work** (2025-11-03, continued session)

After the initial design session, implementation proceeded:

#### **1. MFGComponents Field Extensions** ✅
Extended `MFGComponents` dataclass with 37 new optional fields to support:
- Neural Network MFG (PINN, Deep BSDE)
- Reinforcement Learning MFG (PPO, Actor-Critic)
- Implicit Geometry (level sets, obstacles, manifolds)
- Adaptive Mesh Refinement (refinement criteria, error estimation)
- Time-Dependent Domains (moving boundaries, dynamic obstacles)
- Multi-Population MFG (multiple interacting populations)

**File**: `mfg_pde/core/mfg_problem.py:109-203`

**Impact**: 100% backward compatible (all fields default to None)

#### **2. MFGComponents Docstring Update** ✅
Updated class docstring to reflect:
- "Environment configuration" framing (not just "custom problem definition")
- Comprehensive capabilities list (standard + 6 advanced categories)
- Clearer purpose and relationship to modular solvers

**File**: `mfg_pde/core/mfg_problem.py:30-67`

#### **3. Validation Design** ✅
Created comprehensive validation specification:
- Category-based validation (9 categories)
- Formulation consistency checks
- Dimension validation
- Helpful warning messages
- Optional strict mode

**File**: `docs/development/design/MFGCOMPONENTS_VALIDATION_DESIGN.md`

**Implementation Status**: Design complete, code pending

#### **4. Builder Functions Design** ✅
Created helper function specification:
- 6 formulation-specific builders
- Composition utilities (`merge_components`, `stochastic_mfg_components`)
- Sensible defaults for each formulation
- Usage examples and testing strategy

**File**: `docs/development/design/MFGCOMPONENTS_BUILDER_DESIGN.md`

**Implementation Status**: Design complete, code pending

### **Updated Implementation Roadmap**

#### **Immediate (Completed)** ✅
- Extended MFGComponents with comprehensive field coverage
- Updated docstring to environment configuration framing
- Validation design specification
- Builder functions design specification

#### **Short Term (Next)**
1. **Implement validation logic**
   - Add `MFGComponents.validate()` method
   - Implement category validation helpers
   - Add unit tests for validation

2. **Implement builder functions**
   - Create `mfg_pde/core/component_builders.py`
   - Implement Phase 1 builders (standard, neural, RL, stochastic)
   - Add unit tests for builders

3. **Update user documentation**
   - Add "Using Component Builders" guide
   - Update examples to show builder usage
   - Add validation examples

#### **Medium Term (Later)**
1. **Phase 2 builders**
   - Network, variational, implicit geometry builders
   - Composition utilities

2. **Dimension-agnostic Hamiltonian interface**
   - Adapter pattern for backward compatibility
   - Support array-valued position and momentum

3. **Hamiltonian abstraction**
   - Implement `BaseHamiltonian` abstract class
   - Concrete implementations (Quadratic, PowerLaw, etc.)

### **Design Completeness**

The MFGComponents infrastructure is now fully specified:

| Aspect | Status |
|:-------|:-------|
| **Field Coverage** | ✅ Complete (all 6 formulations) |
| **Docstring** | ✅ Updated (environment framing) |
| **Validation Design** | ✅ Complete (9 categories) |
| **Builder Design** | ✅ Complete (6 builders) |
| **Validation Implementation** | 📝 Pending |
| **Builder Implementation** | 📝 Pending |
| **User Documentation** | 📝 Pending |

---

**Last Updated**: 2025-11-03 (extended session)
**Participants**: Design session with strategic direction setting + implementation
**Next Review**: After validation and builder implementation complete
